import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModel, AutoModelForMaskedLM
import wandb
from tqdm import tqdm
import hydra
from dataset import RNASeq
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.lstm import LSTMClassifier
from modules.transformer import Transformer
import pandas as pd

WANDB_ON = True
MEAN = 0.3998

class_values = torch.tensor([1969, 2962, 4119, 1329, 222])


def mae(y, target): return torch.abs(y-target).mean() - (2 * (F.sigmoid(y).std() - 0.15))
def mae_raw(y, target): return torch.abs(y-target).mean()
def mae_weighted(y, target): return (2*(torch.abs(target-MEAN)+1)*torch.abs(y-target)).mean()
def mae_noreduce(y, target): return torch.abs(y-target)

def train(config):
    
    if WANDB_ON: wandb.login(key="b2e79ea06ca3e1963c1b930a9944bce6938bbb59")
    if WANDB_ON: wandb.init(project=config.log.project, name=config.log.name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set up model
    dna_fm = AutoModelForMaskedLM.from_pretrained(config.dna_fm_id, trust_remote_code=True).to(device)
    if config.model_type == "lstm":
        model = LSTMClassifier(**config.model).to(device)
    elif config.model_type == "transformer":
        model = Transformer(**config.model).to(device)
    else: raise NotImplementedError()
    
    if WANDB_ON: wandb.watch(model)
    
    # setting up data
    dataset = RNASeq(tokenizer_path=config.dna_fm_id, **config.data)
    
    lendf = len(dataset.train)
    
    # class_weights = lendf/(class_values*5)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    # setting up optim
    opt = optim.AdamW(
        model.parameters(), 
        lr=config.train.learning_rate, 
        betas=(config.train.beta1, config.train.beta2), 
        weight_decay=config.train.weight_decay
    )
    
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.25, end_factor=1, total_iters=300)
    cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 75)
    
    
    for epoch in range(config.train.n_epochs):
        train_dl = dataset.train_dataloader()
        
        # training loop run
        opt.zero_grad()
        for i, data in tqdm(enumerate(train_dl), desc=f"Train epoch {epoch}: ", total=len(dataset.train)//dataset.batch_size):
            x, target, att = data
            # preprocess
            # with torch.no_grad():
            #     logits = dna_fm(input_ids=x.to(device), attention_mask=att.to(device), output_hidden_states=True)['hidden_states'][-1] # (batch x seqlen x hidden_dim)
            # print(logits.size)
            # run lstm model
            y = model(x.to(device), att.to(device)).squeeze()
            print(y)
            # print(target)
            
            # compute model loss 
            loss = loss_fn(y, target.to(device)) - (0.5 * (F.sigmoid(y).std() - 0.15))
            print(loss)
            loss.backward()
            if WANDB_ON: wandb.log({"train_loss": loss.item(), "avg_pred": F.sigmoid(y).mean().item(), "avg_gt": target.mean().item(), "epoch": epoch, "pred_std": F.sigmoid(y).std(), "gt_std": target.std(), "lr": opt.param_groups[0]["lr"]})
            # if WANDB_ON: wandb.log({"train_loss": loss.item(), "epoch": epoch})
            
            # run gradient update based on gradient accumulation value
            if i % config.train.grad_accum_iter == 0:
                print("step!")
                # nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip) # clip gradients
                opt.step()
                opt.zero_grad()
                warmup.step()
                cosine.step()
        
        # validation loop run
        val_dl = dataset.val_dataloader()
        
        # disable gradient processing
        with torch.no_grad():
            for i, data in tqdm(enumerate(val_dl), desc=f"Val epoch {epoch}: ", total=len(dataset.val)//dataset.batch_size):
                x, target, att = data
                # compute output
                # logits = dna_fm(input_ids=x.to(device), attention_mask=att.to(device), output_hidden_states=True)['hidden_states'][-1] # (batch x seqlen x hidden_dim)
                y = model(x.to(device), att.to(device)).squeeze()
                
                # compute loss
                loss = loss_fn(y, target.to(device))
                if WANDB_ON: wandb.log({"val_loss": loss.item(), "epoch": epoch})
                
        torch.save(model.state_dict(), f"out/model5_ckpt_{epoch}.pt")
                
    # save model
    torch.save(model.state_dict(), f"{config.log.ckpt_path}/model5.pt")


@torch.no_grad()
def test_regression(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set up model
    dna_fm = AutoModel.from_pretrained(config.dna_fm_id, trust_remote_code=True).to(device)
    if config.model_type == "lstm":
        model = LSTMClassifier(**config.model).to(device)
    elif config.model_type == "transformer":
        model = Transformer(**config.model).to(device)
    else: raise NotImplementedError()
    
    model.load_state_dict(torch.load(config.load_path))
    
    # setting up data
    dataset = RNASeq(tokenizer_path=config.dna_fm_id, **config.data)
        
    test_dl = dataset.test_dataloader()
    
    ys = []
    targets = []
    losses = []
    
    for i, data in tqdm(enumerate(test_dl), desc=f"Evaluating model: ", total=len(dataset.test)//dataset.batch_size):
        x, target, att = data
        # compute output
        logits = dna_fm(input_ids=x.to(device), attention_mask=att.to(device))[0] # (batch x seqlen x hidden_dim)
        y = model(logits, att.to(device)).squeeze()
        
        # compute loss
        loss = mae_noreduce(y, target.to(device))
        
        ys += y.tolist()
        targets += target.tolist()
        losses += loss.tolist()
        
    df = pd.DataFrame({"Y": ys, "Target": targets, "Loss": losses})
    
    df.to_csv(config.out_name)


@torch.no_grad()
def test_classification(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set up model
    dna_fm = AutoModel.from_pretrained(config.dna_fm_id, trust_remote_code=True).to(device)
    if config.model_type == "lstm":
        model = LSTMClassifier(**config.model).to(device)
    elif config.model_type == "transformer":
        model = Transformer(**config.model).to(device)
    else: raise NotImplementedError()
    
    model.load_state_dict(torch.load(config.load_path))
    
    # setting up data
    dataset = RNASeq(tokenizer_path=config.dna_fm_id, **config.data)
        
    test_dl = dataset.test_dataloader()
    
    ys = []
    targets = []
    losses = []
    
    for i, data in tqdm(enumerate(test_dl), desc=f"Evaluating model: ", total=len(dataset.test)//dataset.batch_size):
        x, target, att = data
        # compute output
        logits = dna_fm(input_ids=x.to(device), attention_mask=att.to(device))[0] # (batch x seqlen x hidden_dim)
        y = model(logits, att.to(device)).squeeze()
        
        loss = mae_noreduce(y, target.to(device))
        
        ys += y.tolist()
        targets += target.tolist()
        losses += loss.tolist()
        
    df = pd.DataFrame({"Y": ys, "Target": targets, "Loss": losses})
    
    df.to_csv(config.out_name)
                

    

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    if config.task == "train":
        train(config)
    elif config.task == "test":
        test_classification(config)
    else: raise NotImplementedError()
    
if __name__ == "__main__":
    main()