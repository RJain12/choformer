# import standard modules; re is used for regex on ln 59
import os
from Bio import SeqIO
from Bio.Seq import Seq
import random
import numpy as np
import re

# create dict with value being a tuple with the codons and their probabilities/frequncies
frequency = {
    "A": (["GCG", "GCA", "GCT", "GCC"], [0.07, 0.23, 0.32, 0.37]),
    "R": (["AGG", "AGA", "CGG", "CGA", "CGT", "CGC"], [0.19, 0.19, 0.19, 0.14, 0.11, 0.18]),
    "N": (["AAT", "AAC"], [0.45, 0.55]),
    "D": (["GAT", "GAC"], [0.47, 0.53]),
    "C": (["TGT", "TGC"], [0.47, 0.53]),
    "*": (["TGA", "TAG", "TAA"], [0.53, 0.22, 0.26]),
    "Q": (["CAG", "CAA"], [0.76, 0.24]),
    "E": (["GAG", "GAA"], [0.59, 0.41]),
    "G": (["GGG", "GGA", "GGT", "GGC"], [0.21, 0.25, 0.20, 0.34]),
    "H": (["CAT", "CAC"], [0.44, 0.56]),
    "I": (["ATA", "ATT", "ATC"], [0.14, 0.35, 0.51]),
    "L": (["TTG", "TTA", "CTG", "CTA", "CTT", "CTC"], [0.14, 0.07, 0.39, 0.08, 0.13, 0.19]),
    "K": (["AAG", "AAA"], [0.61, 0.39]),
    "M": (["ATG"], [1.0]),
    "F": (["TTT", "TTC"], [0.47, 0.53]),
    "P": (["CCG", "CCA", "CCT", "CCC"], [0.08, 0.29, 0.31, 0.32]),
    "S": (["AGT", "AGC", "TCG", "TCA", "TCT", "TCC"], [0.15, 0.22, 0.05, 0.14, 0.22, 0.22]),
    "T": (["ACG", "ACA", "ACT", "ACC"], [0.08, 0.29, 0.26, 0.37]),
    "W": (["TGG"], [1.0]),
    "Y": (["TAT", "TAC"], [0.44, 0.56]),
    "V": (["GTG", "GTA", "GTT", "GTC"], [0.46, 0.12, 0.18, 0.24])
}

# Amino acid sequence dir to optimize:
# hardcoded path
aa_dir = os.path.join(os.getcwd(), 'benchmark_sequences', 'aa')

# Output dir to store optimized seqs:
# hardcoded path
out_dir = os.path.join(os.getcwd(), 'benchmark_sequences', 'BFC')


# Normalize probabilities for frequency if sum is not exactly 1.
def fix_p(p):
    if p.sum() != 1.0:
        p = p*(1./p.sum())
    return p


for entry in os.scandir(aa_dir):
    name = entry.name.replace(".fasta", "_dna")

    # Replace ambiguities with amino acids from IUPAC guidelines: https://www.bioinformatics.org/sms/iupac.html
    record = SeqIO.read(entry, "fasta")
    seq = record.seq.replace("B", random.choice(["D", "N"])).replace(
        "Z", random.choice(["E", "Q"]))
    seq_arr = []
    for aa in seq:
        # append to the array a random choice of codon using the probabilities given (p)
        seq_arr.append(np.random.choice(
            frequency[aa][0], p=fix_p(np.asarray(frequency[aa][1]))))

    record.seq = Seq(re.sub('[^GATC]', "", str("".join(seq_arr)).upper()))
    complete_name = os.path.join(out_dir, name)
    SeqIO.write(record, complete_name + ".fasta", "fasta")
