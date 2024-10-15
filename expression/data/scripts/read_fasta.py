from Bio import SeqIO
import csv

fasta_file = "../data_preprocessing/output_clusters.fa"

data = []

with open(fasta_file, "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        gene_id = record.description.split("[")[2].split("]")[0].split("=")[1]
        data.append([str(record.seq), len(record.seq), gene_id, record.id])


with open("dna.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Sequence", "Length", "ID", "full_ID"])
    for i in data:
        writer.writerow(i)