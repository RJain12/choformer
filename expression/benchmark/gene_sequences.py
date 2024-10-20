from Bio import Entrez, SeqIO

Entrez.email = "rishab@samyakscience.com"

def fetch_protein_sequence(accession_id):
    # Search for the protein sequence by accession ID
    try:
        handle = Entrez.efetch(db="protein", id=accession_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        return record.seq
    except Exception as e:
        print(f"Error fetching {accession_id}: {e}")
        return None

# Example accession IDs from your data
accession_ids = ["XP_003507519.1", "XP_003498421.1", "XP_003494975.1", "XP_003501176.1", "XP_003502992.1", "XP_003500393.1"]

# Fetch sequences
for acc_id in accession_ids:
    sequence = fetch_protein_sequence(acc_id)
    if sequence:
        print(f"Accession ID: {acc_id}")
        print(f"Sequence: {sequence}\n")
