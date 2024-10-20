import pandas as pd

file_path = 'inference/cho_wetlab_prot.csv'

data = pd.read_csv(file_path)

protein_sequences_list = data.iloc[:, 0].tolist()

print(protein_sequences_list[:5])
output_file = 'inference/protein_sequences.txt'
with open(output_file, 'w') as file:
    for sequence in protein_sequences_list:
        file.write(sequence + '\n')

print(f"Protein sequences saved to {output_file}")
