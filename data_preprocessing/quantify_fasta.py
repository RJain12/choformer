import math

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

def count_sequences_in_ranges(base_pair_counts):
    ranges = {}
    
    for count in base_pair_counts:
        if count == 0:
            upper_bound = 5
        else:
            power = math.ceil(math.log(count, 5)) 
            upper_bound = 5 ** power
        
        lower_bound = upper_bound // 5
        
        range_label = f"{lower_bound} - {upper_bound}"
        
        # Increment the count for that range
        if range_label not in ranges:
            ranges[range_label] = 0
        ranges[range_label] += 1

    return ranges

def print_ranges_count(ranges):
    for range_label, count in sorted(ranges.items(), key=lambda x: int(x[0].split(' ')[0])):
        print(f"{range_label}: {count} sequences")

def main():
    fasta_file = 'gene.fasta' 
    sequences = parse_fasta(fasta_file)
    base_pair_counts = get_base_pair_counts(sequences)
    ranges = count_sequences_in_ranges(base_pair_counts)
    print_ranges_count(ranges)

if __name__ == '__main__':
    main()
