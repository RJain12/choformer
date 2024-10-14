import os

def split_fasta(input_file, output_dir, num_parts=10):
    with open(input_file, 'r') as fasta_file:
        content = fasta_file.read()

    sequences = content.strip().split('>')[1:]  
    sequences = ['>' + seq for seq in sequences] 

    total_sequences = len(sequences)
    sequences_per_file = total_sequences // num_parts
    remainder = total_sequences % num_parts

    
    start_idx = 0
    for i in range(num_parts):
        
        end_idx = start_idx + sequences_per_file + (1 if i < remainder else 0)
        part_sequences = sequences[start_idx:end_idx]

        output_file = os.path.join(output_dir, f'gene_part_{i + 1}.fasta')
        with open(output_file, 'w') as out_fasta:
            out_fasta.write(''.join(part_sequences))

        start_idx = end_idx

input_fasta = 'gene.fasta'
output_directory = 'output_parts'
os.makedirs(output_directory, exist_ok=True)
split_fasta(input_fasta, output_directory)
