# import standard modules; re is used for regex on ln 59
import os
from Bio import SeqIO
from Bio.Seq import Seq
import random
import numpy as np
import re

# frequencies are from https://www.genscript.com/tools/codon-frequency-table
# create dict with value being a tuple with the codons and their probabilities/frequncies
frequency = {
    "A": "GCC",
    "R": "CGG",
    "N": "AAC",
    "D": "GAC",
    "C": "TGC",
    "*": "TAA",
    "Q": "CAG",
    "E": "GAG",
    "G": "GGC",
    "H": "CAC",
    "I": "ATC",
    "L": "CTG",
    "K": "AAG",
    "M": "ATG",
    "F": "TTC",
    "P": "CCC",
    "S": "AGC",
    "T": "ACC",
    "W": "TGG",
    "Y": "TAC",
    "V": "GTG"
}

# Amino acid sequence dir to optimize:
# hardcoded path
aa_dir = os.path.join(os.getcwd(), 'benchmark_sequences', 'aa')

# Output dir to store optimized seqs:
# hardcoded path
out_dir = os.path.join(os.getcwd(), 'benchmark_sequences', 'HFC')

for entry in os.scandir(aa_dir):
    name = entry.name.replace(".fasta", "_dna")

    # Replace ambiguities with amino acids from IUPAC guidelines: https://www.bioinformatics.org/sms/iupac.html
    record = SeqIO.read(entry, "fasta")
    seq = record.seq.replace("B", random.choice(["D", "N"])).replace(
        "Z", random.choice(["E", "Q"]))
    seq_arr = []
    for aa in seq:
        # append to the array a random choice of codon using the probabilities given (p)
        seq_arr.append(frequency[aa])

    record.seq = Seq(re.sub('[^GATC]', "", str("".join(seq_arr)).upper()))
    complete_name = os.path.join(out_dir, name)
    SeqIO.write(record, complete_name + ".fasta", "fasta")