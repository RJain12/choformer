import pandas as pd
from Bio.Seq import Seq


sequences = pd.read_csv("/Users/Shrey/Shrey_Work/Duke/!Research/ChoFormer/output.csv")
sequences['cleaned_sequence'] = ''

tot = 0

start_codons = ["ATG"]
stop_codons = ["TAA", "TGA", "TAG"]

for idx, row in sequences.iterrows():
    sequence = row['sequence']

    try:
        translated_sequence = Seq(sequence).translate()
    except:
        continue

    if "M" not in translated_sequence:
        continue
    
    try:
        start_idx = next(i for i in range(len(sequence)) if sequence.startswith(tuple(start_codons), i))
    except StopIteration:
        continue

    stop_idxs = [sequence.find(stop_codon) for stop_codon in stop_codons]
    stop_idxs = [idx for idx in stop_idxs if idx != -1]
    if not stop_idxs:
        continue
    stop_idx = min(stop_idxs)


    if stop_idx <= start_idx:
        continue 

    cleaned_sequence = sequence[start_idx:stop_idx+3]
    tot += 1
    sequences.at[idx, 'cleaned_sequence'] = cleaned_sequence

print(sequences)
print(tot)
sequences.to_csv("/Users/Shrey/Shrey_Work/Duke/!Research/ChoFormer/cleaned.csv", index=False)


# # Check if the sequences were cleaned properly
# sequences = pd.read_csv("/Users/Shrey/Shrey_Work/Duke/!Research/ChoFormer/cleaned.csv")
# print(sequences)