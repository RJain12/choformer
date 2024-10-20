import torch
import sys
import torch.nn as nn
from omegaconf import OmegaConf
from dataloader import get_dataloaders
from tqdm import tqdm
from esm_utils import ESMWrapper
from model import DNADecoder
import wandb
from torch.nn.utils.rnn import pad_sequence


def train(train_loader, val_loader, num_epochs, optimizer, device, choformer_model, ckpt_path, config):
    
    wandb.login(key="b2e79ea06ca3e1963c1b930a9944bce6938bbb59")
    wandb.init(project="choformer", name=f"transformer-layer_{config.decoder_model.layers}-heads_{config.decoder_model.heads}-dim_{config.decoder_model.decoder_size}")
    
    best_val_loss = float('inf')
    total_loss = 0
    total_ppl = 0

    # @ Vishrut initialize your wandb for logging loss and perplexity

    ################## TRAINING LOOP ##################
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch+1}/{num_epochs}")
        
        choformer_model.train()

        train_bar = tqdm(total=len(train_loader), leave=True, file=sys.stdout, desc=F"TRAINING EPOCH {epoch+1}")

        for batch in train_loader:
            protein_embeddings, dna_tokens, true_exp = batch
            protein_embeddings, true_exp = protein_embeddings.to(device), true_exp.to(device)
            dna_tokens = torch.stack(dna_tokens).squeeze(1).to(device)

            # ignore loss on first token generation – standard autoregressive implementation
            # dna_tokens[:, 0] = 0

            # zero gradients
            optimizer.zero_grad()

            # Decoder step – use original DNA tokens as true seq for loss calc
            outputs = choformer_model.generate(protein_embeddings, labels=dna_tokens)

            loss = outputs['loss']
            ppl = torch.exp(loss)
            total_loss += loss.item()
            total_ppl += ppl.item()
            
            wandb.log({"epoch": epoch+1, "train_loss": loss.item(), "train_ppl": ppl.item(), "hamming": outputs["hamming"].item()})

            loss.backward()
            optimizer.step()

            train_bar.update(1)
            sys.stdout.flush()
        
        ################## VALIDIATION LOOP ##################
        if val_loader:
            val_loss, val_ppl = _validation(val_loader,choformer_model, device)
            wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_ppl": val_ppl})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(choformer_model.state_dict(), f'{ckpt_path}/best_model.pth')


        ################## END OF TRAINING EPOCH ##################
        # log train and val results
        avg_train_loss = total_loss / len(train_loader)
        avg_train_ppl = total_ppl / len(train_loader)
    
        torch.save(choformer_model.state_dict(), f'{ckpt_path}/epoch_{epoch+1}.pth')
    
    train_bar.close()


def _validation(val_loader, choformer_model, device):
    """Helper method to perform a validation epoch on the choformer model being trained"""
    choformer_model.eval()
    total_val_loss = 0
    total_val_ppl = 0
    
    val_bar = tqdm(total=len(val_loader), leave=True, file=sys.stdout, desc="VALIDATION")

    with torch.no_grad():
        for batch in val_loader:
            protein_embeddings, dna_tokens, true_exp = batch
            protein_embeddings, true_exp = protein_embeddings.to(device), true_exp.to(device)
            dna_tokens = torch.stack(dna_tokens).squeeze(1).to(device)

            # dna_tokens[:, 0] = 0

            outputs = choformer_model.generate(protein_embeddings, labels=dna_tokens)

            loss = outputs['loss']
            ppl = torch.exp(loss)
            total_val_loss += loss.item()
            total_val_ppl += ppl.item()

            val_bar.update(1)
            sys.stdout.flush()
        
    val_bar.close()
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_ppl = total_val_ppl / len(val_loader)

    return avg_val_loss, avg_val_ppl


def test(test_loader, choformer_model, device):
    choformer_model.eval()
    test_loss = 0
    test_ppl = 0

    test_bar = tqdm(total=len(test_loader), leave=True, file=sys.stdout, desc=f"TEST SET")

    with torch.no_grad():
        for batch in test_loader:
            protein_embeddings, dna_tokens, true_exp = batch
            protein_embeddings, true_exp = protein_embeddings.to(device), true_exp.to(device)
            dna_tokens = torch.stack(dna_tokens).squeeze(1).to(device)

            # dna_tokens[:, 0] = 0

            outputs = choformer_model.generate(protein_embeddings, labels=dna_tokens)

            loss = outputs['loss']
            ppl = torch.exp(loss)
            test_loss += loss.item()
            test_ppl += ppl.item()

            test_bar.update(1)
            sys.stdout.flush()
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_ppl = test_ppl / len(test_loader)
    print(f'TEST LOSS: {avg_test_loss}')
    print(f'TEST PERPLEXITY: {avg_test_ppl}')

    print(f"Generated sequences: {outputs['generated_sequences']}")
    #print(f"Predicted expressions: {outputs['predicted_expression']}")


def main(config_path):
    config = OmegaConf.load(config_path)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # instantiate ESM and CHOFormer models
    #esm_model = ESMWrapper()
    choformer_model = DNADecoder(config).to(device)

    # GPU and optimizer
    optim = torch.optim.AdamW(choformer_model.parameters(),
                              lr=config.decoder_hparams.lr,
                              betas=(config.decoder_hparams.beta1, config.decoder_hparams.beta2))

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Train the model
    train(train_loader, val_loader, config.decoder_hparams.num_epochs, optim, device, choformer_model, config.log.ckpt_path, config)
    test(test_loader, choformer_model, device)


if __name__ == "__main__":
    main(config_path="./config.yaml")