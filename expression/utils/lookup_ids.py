import os
import pandas as pd
import time
import random
from Bio import Entrez
from requests.exceptions import HTTPError

# List of random emails to use
email_list = [
    "rishabjain1@college.harvard.edu",
    "rishabjain2@college.harvard.edu",
    "rishabjain3@college.harvard.edu",
    "rishabjain4@college.harvard.edu"
]

def fetch_fasta(id_list):
    # Select a random email from the list
    Entrez.email = random.choice(email_list)
    print(f"Using email: {Entrez.email}")
    
    ids = ",".join(id_list)
    try:
        handle = Entrez.efetch(db="nuccore", id=ids, rettype="fasta", retmode="text")
        fasta_data = handle.read()
        handle.close()
        print('Fetched fasta data for id: ' + ids)
        return fasta_data
    except HTTPError as e:
        if e.response.status_code == 429:  # HTTP 429 Too Many Requests
            print("Rate limit exceeded, sleeping for 5 seconds...")
            time.sleep(5)
            return fetch_fasta(id_list)  # Retry the request
        else:
            raise e

rna = pd.read_csv(os.path.join('..', 'data', 'rna.csv'))
lookup = pd.read_csv(os.path.join('..', 'data', 'rna_lookup.txt'), sep='\t')

for index, row in rna.iterrows():
    id = row['id']
    print(id)
    lookup_row = lookup[lookup['ID'] == id]
    print(lookup_row)
    gb = lookup_row['GB_ACC']
    print(gb)
    print('Looking up fasta data for id: ' + id)

    if not gb.empty:
        fasta_sequences = fetch_fasta(gb)
        rna.at[index, 'fasta'] = fasta_sequences
        print('Fasta data found for id: ' + id)
        
        # Sleep to avoid overwhelming the server
        time.sleep(1)  # Wait 1 second before making another request
    else:
        print("No spot ID found for id: " + id)

rna.to_csv(os.path.join('..', 'data', 'rna_with_fasta.csv'))