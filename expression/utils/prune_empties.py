# go through the rna_expression_seqs folder which is one folder above / expression/rna/rna_expression_seqs

import os
directory = os.path.join('..', 'data', 'rna', 'rna_expression_seqs')

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    
    if os.path.getsize(filepath) == 0:
        os.remove(filepath)
        print(f"Removed empty file: {filename}")