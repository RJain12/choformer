import matplotlib.pyplot as plt

def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        sequence = ''
        for line in f:
            if line.startswith('>'):
                if sequence:  
                    sequences.append(sequence)
                sequence = ''  
            else:
                sequence += line.strip() 
        if sequence:  
            sequences.append(sequence)
    return sequences

def get_base_pair_counts(sequences):
    return [len(seq) for seq in sequences]

def plot_histogram(base_pair_counts):
    sorted_counts = sorted(base_pair_counts)
    
    # X-axis is sequence indices (number of sequences)
    x_values = range(1, len(sorted_counts) + 1)
    
    # Y-axis is base pair counts
    plt.plot(x_values, sorted_counts, marker='o', linestyle='-')
    plt.xlabel('Number of Sequences')
    plt.ylabel('Base Pairs')
    plt.title('Base Pairs per Sequence')
    plt.show()

def main():
    fasta_file = 'gene.fasta'
    sequences = parse_fasta(fasta_file)
    base_pair_counts = get_base_pair_counts(sequences)
    plot_histogram(base_pair_counts)

if __name__ == '__main__':
    main()
