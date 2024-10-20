import torch

# vocab for 3-mer DNA tokenizer
vocab = {"[CLS]": 0, "[EOS]": 1, "[PAD]": 2}
AGCT = {"A": 0, "G": 1, "C": 2, "T": 3, "N": 4}

def process_codon(seq: str):
    try:
        idx_1 = AGCT[seq[0]]
        idx_2 = AGCT[seq[1]]
        idx_3 = AGCT[seq[2]]
        return 25 * idx_1 + 5 * idx_2 + idx_3 + 3
    except:
        return 1  # return a default index for invalid codons

def encode(sequences: list[str], max_length: int):
    inputs = []
    
    for seq in sequences:
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        tokens = [0, *[process_codon(codon) for codon in codons]]
        # Ensure tokens do not exceed max_length
        if len(tokens) >= max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [2] * (max_length - len(tokens))  # Pad with [PAD] index
        
        inputs.append(torch.tensor(tokens))
    
    return torch.stack(inputs, dim=0)


def decode(ids: torch.Tensor):
    sequences = []
    for token_ids in ids:
        codons = []
        for token_id in token_ids:
            if token_id == 2:  # PAD token
                break
            if token_id not in [0, 1]:  # Skip CLS and EOS tokens
                codon = token_id_to_codon(token_id)
                if codon:
                    codons.append(codon)
                else:
                    codons.append("XXX")
        
        full_sequence = ''.join(codons)
        sequences.append(full_sequence)

    return sequences

def token_id_to_codon(token_id):
    rev_acgt = {v: k for k, v in AGCT.items()}
    # Decode the codon from the token ID
    if token_id < 3:  # Ignore special tokens
        return None
    # Calculate the codon indices from the token ID
    token_id = token_id.item() - 3  # Subtract 3 to ignore special tokens and adjust for 0-indexing
    idx_1 = token_id // 25
    idx_2 = (token_id % 25) // 5
    idx_3 = token_id % 5
    # Map back to the nucleotide
    return rev_acgt[idx_1] + rev_acgt[idx_2] + rev_acgt[idx_3]