import torch
import sys
import torch.nn as nn
from omegaconf import OmegaConf
from dataloader import get_dataloaders
from tqdm import tqdm
from esm_utils import ESMWrapper
from model import DNADecoder
from torch.nn.utils.rnn import pad_sequence


def train(train_loader, val_loader, num_epochs, optimizer, device, choformer_model, ckpt_path):
    
    best_val_loss = float('inf')
    total_loss = 0
    total_ppl = 0

    # @ Vishrut initialize your wandb for logging loss and perplexity

    ################## TRAINING LOOP ##################
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch+1}/{num_epochs}")
        
        choformer_model.train()

        train_bar = tqdm(total=len(train_loader), leave=True, file=sys.stdout, desc=F"TRAINING EPOCH: {epoch+1}")

        for batch in train_loader:
            protein_embeddings, dna_tokens, true_exp = batch
            protein_embeddings, true_exp = protein_embeddings.to(device), true_exp.to(device)

            dna_tokens = [dct['input_ids'].squeeze(0).to(device) for dct in dna_tokens]
            dna_tokens = pad_sequence(dna_tokens,
                                      batch_first=True,
                                      padding_value=choformer_model.dna_tokenizer.pad_token_id)

            # ignore loss on first token generation – standard autoregressive implementation
            dna_tokens[:, 0] = -100

            # zero gradients
            optimizer.zero_grad()

            # Decoder step – use original DNA tokens as true seq for loss calc
            outputs = choformer_model.generate(protein_embeddings, labels=dna_tokens)

            loss = outputs.loss
            ppl = torch.exp(loss / torch.where(dna_tokens != choformer_model.dna_tokenizer.pad_token_id).sum().item())
            total_loss += loss.item()
            total_ppl += ppl.item()

            loss.backward()
            optimizer.step()

            train_bar.update(1)
            sys.stdout.flush()
        
        ################## VALIDIATION LOOP ##################
        if val_loader:
            val_loss, val_ppl = _validation(val_loader,choformer_model, device)
            #wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_ppl": val_ppl})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(choformer_model.state_dict(), f'{ckpt_path}/best_model_epoch{epoch+1}.pth')


        ################## END OF TRAINING EPOCH ##################
        # log train and val results
        avg_train_loss = total_loss / len(train_loader)
        avg_train_ppl = total_ppl / len(train_loader)
        #wandb.log({"epoch": {epoch+1}, "train_loss": avg_train_loss, "train_ppl": avg_train_ppl})
    
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

            dna_tokens[:, 0] = -100

            outputs = choformer_model.generate(protein_embeddings, labels=dna_tokens)

            loss = outputs.loss
            ppl = torch.exp(loss / torch.where(dna_tokens != choformer_model.dna_tokenizer.pad_token_id).sum().item())
            total_val_loss += loss.item()
            total_val_ppl += ppl.item()

            val_bar.update(1)
            sys.stdout.flush()
        
    val_bar.close()
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_ppl = total_val_ppl / len(val_loader)

    return avg_val_loss, avg_val_ppl


def test(test_loader, choformer_model, esm_model, device):
    choformer_model.eval()
    test_loss = 0
    test_ppl = 0

    test_bar = tqdm(total=len(test_loader), leave=True, file=sys.stdout, desc=f"TEST SET")

    with torch.no_grad():
        generated_sequences = []
        for batch in test_loader:
            protein_embeddings, dna_tokens, true_exp = batch
            protein_embeddings, dna_tokens, true_exp = protein_embeddings.to(device), dna_tokens.to(device)['input_ids'], true_exp.to(device)

            dna_tokens[:, 0] = -100

            outputs = choformer_model.generate(protein_embeddings, labels=dna_tokens)
            generated_sequences.append(outputs['generated_sequences'])

            loss = outputs.loss
            ppl = torch.exp(loss / torch.where(dna_tokens != choformer_model.dna_tokenizer.pad_token_id).sum().item())
            test_loss += loss.item()
            test_ppl += ppl.item()

            test_bar.update(1)
            sys.stdout.flush()
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_ppl = test_ppl / len(test_loader)
    print(f'TEST LOSS: {avg_test_loss}')
    print(f'TEST PERPLEXITY: {avg_test_ppl}')


def main(config_path):
    config = OmegaConf.load(config_path)

    # instantiate ESM and CHOFormer models
    #esm_model = ESMWrapper()
    choformer_model = DNADecoder(config)

    # GPU and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.AdamW(choformer_model.parameters(),
                              lr=config.decoder_hparams.lr,
                              betas=(config.decoder_hparams.beta1, config.decoder_hparams.beta2))

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Train the model
    train(train_loader, val_loader, config.decoder_hparams.num_epochs, optim, device, choformer_model, config.log.ckpt_path)
    test(test_loader, choformer_model, device)


if __name__ == "__main__":
    main(config_path="./config.yaml")