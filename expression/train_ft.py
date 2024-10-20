import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModel
import wandb
from tqdm import tqdm
import hydra
from dataset import RNASeq
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.lstm import LSTMClassifier
from modules.transformer import Transformer

WANDB_ON = False

hidden_size = 768

class DNAFMClassifier(nn.Module):
    def __init__(self, dna_fm_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(dna_fm_id, trust_remote_code=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        out = F.sigmoid(self.classifier(self.dropout(out)))
        return out
         

def mae(y, target): return torch.abs(y-target).mean()

def train(config):
    
    if WANDB_ON: wandb.login(key="b2e79ea06ca3e1963c1b930a9944bce6938bbb59")
    if WANDB_ON: wandb.init(project=config.log.project, name=config.log.name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set up model
    dna_fm = DNAFMClassifier(config.dna_fm_id).to(device)
    # if config.model_type == "lstm":
    #     model = LSTMClassifier(**config.model).to(device)
    # elif config.model_type == "transformer":
    #     model = Transformer(**config.model).to(device)
    # else: raise NotImplementedError()
    
    if WANDB_ON: wandb.watch(dna_fm)
    
    # setting up data
    dataset = RNASeq(tokenizer_path=config.dna_fm_id, **config.data)
    
    # setting up optim
    opt = optim.AdamW(
        dna_fm.parameters(), 
        lr=config.train.learning_rate, 
        betas=(config.train.beta1, config.train.beta2), 
        weight_decay=config.train.weight_decay
    )
    
    
    for epoch in range(config.train.n_epochs):
        train_dl = dataset.train_dataloader()
        
        # training loop run
        opt.zero_grad()
        for i, data in tqdm(enumerate(train_dl), desc=f"Train epoch {epoch}: ", total=len(dataset.train)//dataset.batch_size):
            x, target, att = data
            # preprocess
            # with torch.no_grad():
            y = dna_fm(input_ids=x.to(device), attention_mask=att.to(device)) # (batch x seqlen x hidden_dim)
            print(y)
            
            # run lstm model
            # y = model(logits, att.to(device)).squeeze()
            
            # compute model loss 
            loss = mae(y, target.to(device))
            print(loss)
            loss.backward()
            if WANDB_ON: wandb.log({"train_loss": loss.item(), "avg_pred": y.mean().item(), "avg_gt": target.mean().item(), "epoch": epoch, "pred_std": y.std(), "gt_std": target.std()})
            
            # run gradient update based on gradient accumulation value
            if i % config.train.grad_accum_iter == 0:
                print("step!")
                nn.utils.clip_grad_norm_(dna_fm.parameters(), config.train.grad_clip) # clip gradients
                opt.step()
                opt.zero_grad()
        
        # validation loop run
        val_dl = dataset.val_dataloader()
        
        # disable gradient processing
        with torch.no_grad():
            for i, data in tqdm(enumerate(val_dl), desc=f"Val epoch {epoch}: ", total=len(dataset.val)//dataset.batch_size):
                x, target, att = data
                # compute output
                y = dna_fm(input_ids=x.to(device), attention_mask=att.to(device)) # (batch x seqlen x hidden_dim)
                # y = model(logits, att.to(device)).squeeze()
                
                # compute loss
                # loss = mae(y, target.to(device))
                if WANDB_ON: wandb.log({"val_loss": loss.item(), "epoch": epoch})
                
    # save model
    torch.save(dna_fm.state_dict(), f"{config.log.ckpt_path}/ft_model.pt")

    

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    if config.task == "train":
        train(config)
    else: raise NotImplementedError()
    
if __name__ == "__main__":
    main()