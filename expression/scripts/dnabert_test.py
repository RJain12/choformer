import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
inv_map = {v: k for k, v in tokenizer.vocab.items()}
print(inv_map[0], inv_map[1], inv_map[2], inv_map[3], inv_map[4], inv_map[5])
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).cuda()

dna = ["ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC", "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGCGTTAGCGTTAGCGTTAGCGTTAGCGTTAGCGTTAGCGTTAGCGTTAGCGTTAGC"]
inputs = tokenizer(dna, return_tensors = 'pt', padding=True)["input_ids"]
print(inputs)
hidden_states = model(inputs.cuda())[0] # [1, sequence_length, 768]

