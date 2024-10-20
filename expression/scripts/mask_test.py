import torch

def create_square_mask(seq_mask):
    """
    Convert sequence-based mask (batch, seq_len) to square mask (batch, 1, seq_len, seq_len)
    suitable for torch.nn.functional.scaled_dot_product_attention.
    """
    # (batch_size, seq_len) -> (batch_size, 1, seq_len, seq_len)
    square_mask = seq_mask.unsqueeze(1) * seq_mask.unsqueeze(2)  # Broadcasting
    return (square_mask + torch.diag(torch.ones(seq_mask.size(-1)))).clamp(0, 1)

# Example sequence-based mask (1 indicates valid positions, 0 indicates padding)
batch_size = 2
seq_len = 4
seq_mask = torch.tensor([[1, 1, 1, 0],   # Mask for the first batch
                         [1, 1, 0, 0]])  # Mask for the second batch

# Convert to square mask
square_mask = create_square_mask(seq_mask)
print(square_mask)
