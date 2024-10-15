import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModel
import wandb
from tqdm import tqdm
import hydra
from dataset import RNASeq


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True) # bidirectional?
        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.Linear(input_size//2, 1),
        )
                
    def forward(self, x: Tensor) -> Tensor:
        b, l, d = x.size()
        
        # cls_tok_expand = self.cls_token.unsqueeze(0).repeat(b, 1).unsqueeze(1)
        # x = torch.concat([cls_tok_expand, x, cls_tok_expand], dim=-2)
        
        logit = self.lstm(x)[0].mean(dim=-2) # NOTE: Idea for pooling: add cls token at beginning and end and add - both final state representations, NVM AVERAGE POOLING!!
        return F.sigmoid(self.classifier(logit)) # sigmoid here for mse comparison


def train(config):
    
    wandb.login(key="b2e79ea06ca3e1963c1b930a9944bce6938bbb59")
    wandb.init(project=config.log.project, name=config.log.name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set up model
    dna_fm = AutoModel.from_pretrained(config.dna_fm_id, trust_remote_code=True).to(device)
    model = LSTMClassifier(**config.model).to(device)
    
    wandb.watch(model)
    
    # setting up data
    dataset = RNASeq(tokenizer_path=config.dna_fm_id, **config.data)
    
    # setting up optim
    opt = optim.AdamW(
        model.parameters(), 
        lr=config.train.learning_rate, 
        betas=(config.train.beta1, config.train.beta2), 
        weight_decay=config.train.weight_decay
    )
    
    
    for epoch in range(config.train.n_epochs):
        train_dl = dataset.train_dataloader()
        
        # training loop run
        opt.zero_grad()
        for i, data in tqdm(enumerate(train_dl), desc=f"Train epoch {epoch}: ", total=len(dataset.train)//dataset.batch_size):
            x, target = data
            # preprocess
            with torch.no_grad():
                logits = dna_fm(input_ids=x.to(device))[0] # (batch x seqlen x hidden_dim)
            
            # run lstm model
            y = model(logits).squeeze()
            
            # compute model loss 
            loss = F.mse_loss(y, target.to(device))
            loss.backward()
            # wandb.log({"train_loss": loss.item()})
            
            # run gradient update based on gradient accumulation value
            if i % config.train.grad_accum_iter == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip) # clip gradients
                opt.step()
                opt.zero_grad()
        
        # validation loop run
        val_dl = dataset.val_dataloader()
        
        # disable gradient processing
        with torch.no_grad():
            for i, data in tqdm(enumerate(val_dl), desc=f"Val epoch {epoch}: ", total=len(dataset.val)//dataset.batch_size):
                x, target = data
                # compute output
                logits = dna_fm(input_ids=x.to(device))[0] # (batch x seqlen x hidden_dim)
                y = model(logits).squeeze()
                
                # compute loss
                loss = F.mse_loss(y, target.to(device))
                # wandb.log({"val_loss": loss.item()})
                
    # save model
    torch.save(model.state_dict(), f"{config.log.ckpt_path}/model.pt")

    

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    if config.task == "train":
        train(config)
    else: raise NotImplementedError()
    
if __name__ == "__main__":
    main()