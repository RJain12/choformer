import torch
import wandb
import sys
import config
from dataloader import get_dataloaders
from tqdm import tqdm
import torch.nn as nn
from esm_utils import ESMWrapper
from model import DNADecoder

def train(train_loader, val_loader, num_epochs, optimizer, device, choformer_model, esm_model, ckpt_path):
    
    best_val_loss = float('inf')
    total_loss = 0
    total_ppl = 0

    # @ Vishrut initialize your wandb for logging loss and perplexity

    train_bar = tqdm(total=len(train_loader), leave=True, file=sys.stdout, desc=F"TRAINING EPOCH: {epoch+1}")

    ################## TRAINING LOOP ##################
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch+1}/{num_epochs}")
        
        choformer_model.train()
        esm_model.eval()

        for batch in train_loader:
            og_sequences, attn_mask, true_exp = batch

            # Encoder step – don't compute gradients for encoder to get protein embeddings
            with torch.no_grad():
                protein_embeddings = esm_model.get_embeddings(og_sequences).to(device)

            # Get tokens for original DNA sequence and clone to use as labels for loss calculation
            input_ids = choformer_model.dna_tokenizer(og_sequences, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
            labels = input_ids.clone()

            # zero gradients
            optimizer.zero_grad()

            # Decoder step
            outputs = choformer_model.generate(protein_embeddings, input_ids, labels=labels)

            loss = outputs.loss
            ppl = torch.exp(loss)
            total_loss += loss.item()
            total_ppl += ppl.item()

            loss.backward()
            optimizer.step()

            train_bar.update(1)
            sys.stdout.flush()
        
        ################## VALIDIATION LOOP ##################
        if val_loader:
            val_loss, val_ppl = _validation(val_loader,choformer_model, esm_model, device)
            wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_ppl": val_ppl})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(choformer_model.state_dict(), f'{ckpt_path}/best_model_epoch{epoch+1}.pth')


        ################## END OF TRAINING EPOCH ##################
        # log train and val results
        avg_train_loss = total_loss / len(train_loader)
        avg_train_ppl = total_ppl / len(train_loader)
        wandb.log({"epoch": {epoch+1}, "train_loss": avg_train_loss, "train_ppl": avg_train_ppl})
    train_bar.close()


def _validation(val_loader, choformer_model, esm_model, device):
    """Helper method to perform a validation epoch on the choformer model being trained"""
    choformer_model.eval()
    total_val_loss = 0
    total_val_ppl = 0
    
    val_bar = tqdm(total=len(val_loader), leave=True, file=sys.stdout, desc="VALIDATION")

    with torch.no_grad():
        for batch in val_loader:
            og_sequences, attn_mask, true_exp = batch
            protein_embeddings = esm_model.get_embeddings(og_sequences).to(device)

            input_ids = choformer_model.dna_tokenizer(og_sequences, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
            labels = input_ids.clone()
            labels[:, 0] = -100 # ignore loss on first (cls) token

            outputs = choformer_model(protein_embeddings, input_ids, labels=labels)

            loss = outputs.loss
            ppl = torch.exp(loss)
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
            og_sequences, attn_mask, expressions = batch
            protein_embeddings = esm_model.get_embeddings(og_sequences).to(device)

            input_ids = choformer_model.dna_tokenizer(og_sequences, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
            labels = input_ids.clone()
            labels[:, 0] = -100

            outputs = choformer_model(protein_embeddings, input_ids, labels=labels)
            generated_sequences.append(outputs['generated_sequences'])

            loss = outputs.loss
            ppl = torch.exp(loss)
            test_loss += loss.item()
            test_ppl += ppl.item()

            test_bar.update(1)
            sys.stdout.flush()
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_ppl = test_ppl / len(test_loader)
    print(f'TEST LOSS: {avg_test_loss}')
    print(f'TEST PERPLEXITY: {avg_test_ppl}')


def main(config):
    # GPU and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.AdamW(choformer_model.parameters(),
                              lr=config.decoder_hparams.lr,
                              betas=(config.decoder_hparams.beta1, config.decoder_hparams.beta2))
    

    # instantiate ESM and CHOFormer models
    esm_model = ESMWrapper()
    choformer_model = DNADecoder(**config.decoder_model)

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Train the model
    train(train_loader, val_loader, config.decoder_hparams.num_epochs, optim, choformer_model, esm_model, config.log.ckpt_path)
    test(test_loader, choformer_model, esm_model, device)


if __name__ == "__main__":
    main(config)
