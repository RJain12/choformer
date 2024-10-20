import io
import csv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.param_functions import Form
from pydantic import BaseModel
import torch
import torch.nn as nn
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

import sys
sys.path.append('C:\\Users\\rkfam\\choformer')


import torch.nn.functional as F
import torch.nn as nn

# from choformer import tokenizer
# from choformer.model import DNADecoder
from inference.model import DNADecoder

from omegaconf import OmegaConf
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from transformers import AutoModel, AutoTokenizer

config = OmegaConf.load("C:\\Users\\rkfam\\choformer\\choformer\\config.yaml")

print(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

choformer_ = DNADecoder(config).to(device).eval()
choformer_.load_state_dict(torch.load("C:\\Users\\rkfam\\choformer\\choformer\\ckpts\\best_model.pth", map_location=device))

esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

def run_choformer_inference(protein_sequences: list[str]):
    print('p', protein_sequences)
    longest_protein_length = max([len(sequence) for sequence in protein_sequences])

    protein_tokens = esm_tokenizer(
        protein_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=longest_protein_length
    ).to(device)


    with torch.no_grad():
        protein_embeddings = esm_model(**protein_tokens).last_hidden_state.squeeze(0)

    print('pp', protein_embeddings)

    outputs = choformer_.generate(protein_embeddings)
    
    return outputs['generated_sequences']


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

class ProteinSequences(BaseModel):
    sequences: list[str]

class GeneratedSequences(BaseModel):
    sequences: list[str]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/choformer_inference/", response_model=GeneratedSequences)
async def choformer_inference(protein_sequences: ProteinSequences):
    try:
        generated_sequences = run_choformer_inference(protein_sequences.sequences)
        return GeneratedSequences(sequences=generated_sequences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_csv")
async def process_csv(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type")
    content = await file.read()
    csv_reader = csv.reader(io.StringIO(content.decode('utf-8')))
    results = []
    for row in csv_reader:
        # result = process_text(row[0])
        result = ''
        results.append(result) 
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
