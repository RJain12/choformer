'''
Generates a sequence of codons and then iterates through the sequence, constantly adjusting the current codon to maximize CAI.
Goal of this is to find a combination of codons to maximize CAI (achieve 1.0 CAI).
'''

# Import modules
import os
from Bio import SeqIO
from Bio.Seq import Seq
import random
import numpy as np
import math
import re

# Set input AA sequence directory and output for writing brute sequences
aa_dir = os.path.join(os.getcwd(), 'benchmark_sequences', 'aa')
out_dir = os.path.join(os.getcwd(), 'benchmark_sequences', 'ERC')

# import the weights from /Users/rishabjain/Desktop/Research/choformer/benchmarking/codon_weights.csv (columns: codon, weight) and store them in a dictionary
weights = {}
with open('/Users/rishabjain/Desktop/Research/choformer/benchmarking/codon_weights.csv', 'r') as f:
    for line in f:
        line = line.strip().split(',')
        # check if line is header line
        if line[0] == 'codon':
            continue
        weights[line[0]] = float(line[1])
        print(weights)

# Create a list of all the codons and match their corresponding weights
codons = []
for key in weights:
    codons.append(key)

def seq2cai(codonarray):
    output = []

    for codon in codonarray:
        output.append(weights[codon])
    length = 1 / len(codonarray)
    return pow(math.prod(output), length)


def aa2codons(seq: str) -> list:
    _aas = {
        "A": ["GCT GCC GCA GCG"],
        "R": ["CGT CGC CGA CGG AGA AGG"],
        "N": ["AAT AAC"],
        "D": ["GAT GAC"],
        "C": ["TGT TGC"],
        "Q": ["CAA CAG"],
        "E": ["GAA GAG"],
        "G": ["GGT GGC GGA GGG"],
        "H": ["CAT CAC"],
        "I": ["ATT ATC ATA"],
        "L": ["TTA TTG CTT CTC CTA CTG"],
        "K": ["AAA AAG"],
        "M": ["ATG ATG"],
        "F": ["TTT TTC"],
        "P": ["CCT CCC CCA CCG"],
        "S": ["TCT TCC TCA TCG AGT AGC"],
        "T": ["ACT ACC ACA ACG"],
        "W": ["TGG TGG"],
        "Y": ["TAT TAC"],
        "V": ["GTT GTC GTA GTG"],
        "B": ["GAT GAC AAT AAC"],
        "Z": ["GAA GAG CAA CAG"],
        "*": ["TAA TAG TGA"],
    }
    return [_aas[i] for i in seq]


# Converts an amino acid to a random corresponding codon:
for entry in os.scandir(aa_dir):
    # Read in the amino acid sequence:
    name = entry.name.replace(".fasta", "_dna")
    record = SeqIO.read(entry, 'fasta')

    masterlist = []
    bestcai = 0
    curcai = 0
    TOTAL_ITERATIONS = 10000

    for curr_iteration in range(0, TOTAL_ITERATIONS):
        codonarr = []
        # Convert amino acid to codons:
        for i in record.seq:
            # Randomly choose a codon from the list of codons for the amino acid:
            codonarr.append(random.choice(aa2codons(i)[0][0].split()))
        masterlist.append(codonarr)
        # With our new codon array, calculate the CAI:
        cai = seq2cai(codonarr)
        if (cai > curcai):
            bestcai = curr_iteration
            curcai = cai
            print('new best cai ' + str(cai))
        curr_iteration += 1
        print(curr_iteration)

    # Write the codon array to a file:
    record.seq = Seq(re.sub('[^GATC]', "", str(
        "".join(masterlist[bestcai])).upper()))
    complete_name = os.path.join(out_dir, name)
    SeqIO.write(record, complete_name + ".fasta", "fasta")
