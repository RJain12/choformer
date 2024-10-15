import pandas as pd
from Bio.Seq import Seq

sequences = pd.read_csv("/Users/Shrey/Shrey_Work/Duke/!Research/ChoFormer/output.csv")
sequences['cleaned_sequence'] = ''

goods = 0
no_trans = 0
no_m = 0
no_start = 0
no_stop = 0

start_codons = ["ATG"]
stop_codons = ["TAA", "TGA", "TAG"]

def find_start_codon(sequence, codon_list):
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in codon_list:
            return i
    return -1

def find_stop_codon(sequence, start_idx, codon_list):
    for i in range(start_idx + 3, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in codon_list:
            return i
    return -1

for idx, row in sequences.iterrows():
    sequence = row['sequence']

    try:
        translated_sequence = str(Seq(sequence).translate())
    except:
        no_trans += 1
        continue

    if "M" not in translated_sequence:
        no_m += 1
        continue

    start_idx = find_start_codon(sequence, codon_list=start_codons)
    if start_idx == -1:
        no_start += 1
        continue

    stop_idx = find_stop_codon(sequence, start_idx, codon_list=stop_codons)
    if stop_idx == -1:
        no_stop += 1
        continue

    cleaned_sequence = sequence[start_idx:stop_idx+3]
    goods += 1
    sequences.at[idx, 'cleaned_sequence'] = cleaned_sequence

print(sequences)

print(f"good sequences: {goods}")
print(f"no start codon: {no_start}")
print(f"no stop after start codon: {no_stop}")
print(f"no met: {no_m}")
sequences.to_csv("/Users/Shrey/Shrey_Work/Duke/!Research/ChoFormer/cleaned.csv", index=False)

# # Check if the sequences were cleaned properly
# sequences = pd.read_csv("/Users/Shrey/Shrey_Work/Duke/!Research/ChoFormer/cleaned.csv")
# print(sequences)