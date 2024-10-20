import csv
from Bio import Entrez, SeqIO

# Set Entrez email for NCBI requests
Entrez.email = "rishab@samyakscience.com"

# Step 1: Fetch Protein Sequence
def fetch_protein_sequence(accession_id):
    """Fetches the protein sequence for a given accession ID."""
    try:
        handle = Entrez.efetch(db="protein", id=accession_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        return str(record.seq)  # Convert Seq object to string
    except Exception as e:
        print(f"Error fetching protein sequence for {accession_id}: {e}")
        return None

# Step 2: Fetch Gene Sequence
def fetch_gene_sequence(accession_id):
    """Fetches the gene sequence based on the protein's CDS coded_by field."""
    try:
        # Fetch the GenBank entry for the protein ID
        handle = Entrez.efetch(db="protein", id=accession_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()

        # Look for the CDS feature and the coded_by qualifier to find the gene accession ID
        for feature in record.features:
            if feature.type == "CDS" and "coded_by" in feature.qualifiers:
                coded_by_info = feature.qualifiers["coded_by"][0]
                gene_accession_id = coded_by_info.split(":")[0]  # Extract gene accession ID
                return fetch_nucleotide_sequence(gene_accession_id)

        return None
    except Exception as e:
        print(f"Error fetching gene sequence for {accession_id}: {e}")
        return None

def fetch_nucleotide_sequence(nucleotide_id):
    """Fetches the nucleotide sequence for a given nucleotide accession ID."""
    try:
        handle = Entrez.efetch(db="nucleotide", id=nucleotide_id, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        return str(record.seq)  # Convert Seq object to string
    except Exception as e:
        print(f"Error fetching nucleotide sequence for {nucleotide_id}: {e}")
        return None

# Input and output file paths
input_file = "processed_proteinexp.csv"
protein_output_file = "proteinseq_proteinexp.csv"
gene_output_file = "geneseq_proteinexp.csv"

# Step 1: Fetch protein sequences and save to proteinseq_proteinexp.csv
with open(input_file, mode="r", newline='', encoding="utf-8") as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Skip the header
    
    with open(protein_output_file, mode="w", newline='', encoding="utf-8") as protein_outfile:
        writer = csv.writer(protein_outfile)
        new_header = ["Protein Sequence"] + header  # Add a new "Protein Sequence" column to the header
        writer.writerow(new_header)

        for row in reader:
            accession_id = row[0]  # Protein Accession ID
            protein_sequence = fetch_protein_sequence(accession_id)

            if protein_sequence:
                writer.writerow([protein_sequence] + row)
            else:
                print(f"Skipping accession {accession_id} due to missing protein sequence.")

# Step 2: Fetch gene sequences and save to geneseq_proteinexp.csv
with open(input_file, mode="r", newline='', encoding="utf-8") as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Skip the header again

    with open(gene_output_file, mode="w", newline='', encoding="utf-8") as gene_outfile:
        writer = csv.writer(gene_outfile)
        new_header = ["Gene Sequence"] + header  # Add a new "Gene Sequence" column to the header
        writer.writerow(new_header)

        for row in reader:
            accession_id = row[0]  # Protein Accession ID
            gene_sequence = fetch_gene_sequence(accession_id)

            if gene_sequence:
                writer.writerow([gene_sequence] + row)
            else:
                print(f"Skipping accession {accession_id} due to missing gene sequence.")
