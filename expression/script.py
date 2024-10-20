import torch
import torch.nn.functional as F
from modules.transformer import Transformer
from dataset import tokenize
import pandas as pd
import hydra
from tqdm import tqdm

PRETRAIN_PATH = "out/model4_ckpt_3.pt"
CHODATA_PATH = "chodata_final.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(config):
    model = Transformer(**config.model).to(device)
    
    df = pd.read_csv(CHODATA_PATH)
    
    outs = []
    
    for dna in tqdm(df["dna"]):
        tokens = tokenize([dna], max_length=1024)
        att = torch.where(tokens!=2, torch.ones_like(tokens), torch.zeros_like(tokens))
        out = F.sigmoid(model(tokens.to(device), att.to(device))).item()
        outs.append(out)
    
    df["expression"] = outs
    
    df.to_csv("chodata_final_exp.csv")

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    evaluate(config)

if __name__ == "__main__":
    main()