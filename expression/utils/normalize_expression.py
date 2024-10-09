import os
import pandas as pd

# get rna.csv
rna = pd.read_csv(os.path.join('..', 'data', 'rna', 'rna.csv'))

# normalize second column values from 0 to 1
rna['expression'] = rna['expression'] / rna['expression'].max()

# save to rna_normalized.csv
rna.to_csv(os.path.join('..', 'data', 'rna', 'rna_normalized.csv'), index=False)