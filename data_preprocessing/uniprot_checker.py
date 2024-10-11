import requests
from Bio import SeqIO

input_file = 'cleaned_protein_sequences.fa'

uniprot_url = "https://rest.uniprot.org/uniprotkb/search?query=sequence:"

def search_protein(sequence):
    """
    search uniprot for protein sequence.
    """
    response = requests.get(f"{uniprot_url}{sequence}&format=json")
    if response.status_code == 200:
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            # Extract protein names and IDs from the results
            return [(result["primaryAccession"], result["proteinDescription"]["recommendedName"]["fullName"]["value"])
                    for result in data["results"]]
        else:
            return []
    else:
        print(f"Error querying UniProt for sequence {sequence}: {response.status_code}")
        return []

def check_sequences(input_file):
    for record in SeqIO.parse(input_file, "fasta"):
        protein_seq = str(record.seq)
        # Search the protein in UniProt
        matches = search_protein(protein_seq)
        if matches:
            print(f"Sequence {record.id} matches the following proteins:")
            for acc, name in matches:
                print(f"- {name} (Accession: {acc})")
        else:
            print(f"No matches found for sequence {record.id}")

check_sequences(input_file)
