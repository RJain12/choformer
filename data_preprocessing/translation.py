from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable

# Amino Acids	Three Letter Code	Single Letter Code
# Glycine	GLY	G
# Alanine	ALA	A
# Valine	VAL	V
# Leucine	LEU	L
# IsoLeucine	ILE	I
# Threonine	THR	T
# Serine	SER	S
# Methionine	MET	M
# Cystein	CYS	C
# Proline	PRO	P
# Phenylalanine	PHE	F
# Tyrosine	TYR	Y
# Tryptophane	TRP	W
# Histidine	HIS	H
# Lysine	LYS	K
# Argenine	ARG	R
# Aspartate	ASP	D
# Glutamate	GLU	E
# Asparagine	ASN	N
# Glutamine	GLN	Q
input_file = 'output_clusters.fa'  
output_file = 'cleaned_protein_sequences.fa' 

natural_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

def clean_and_translate_sequences(input_file, output_file):
    with open(output_file, 'w') as output_handle:
        # Parse the input FASTA file
        for record in SeqIO.parse(input_file, "fasta"):
            try:
                # Remove any non-standard characters from the nucleotide sequence
                cleaned_seq = ''.join(filter(lambda x: x in 'ATCGatcg', str(record.seq)))
                seq_obj = Seq(cleaned_seq)

                # Translate the nucleotide sequence into a protein sequence
                protein_seq = seq_obj.translate(table="Standard", to_stop=True)

                # Check if the translated protein sequence contains unnatural amino acids
                if all(aa in natural_amino_acids for aa in protein_seq):
                    # Update the SeqRecord with the translated protein sequence
                    record.seq = protein_seq
                    # Write the cleaned protein sequence to the output FASTA file
                    SeqIO.write(record, output_handle, "fasta")

            except CodonTable.TranslationError as e:
                # Handle codon translation issues (e.g., incomplete codons)
                print(f"Translation error for {record.id}: {e}")
            except Exception as e:
                # General error handling
                print(f"An error occurred with record {record.id}: {e}")

# Run the cleaning and translation process
clean_and_translate_sequences(input_file, output_file)
