import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


def create_square_mask(seq_mask):
    """
    Convert sequence-based mask (batch, seq_len) to square mask (batch, 1, seq_len, seq_len)
    suitable for torch.nn.functional.scaled_dot_product_attention.
    """
    # (batch_size, seq_len) -> (batch_size, 1, seq_len, seq_len)
    square_mask = seq_mask.unsqueeze(1) * seq_mask.unsqueeze(2)  # Broadcasting
    return (square_mask + torch.diag(torch.ones(seq_mask.size(-1))).cuda()).clamp(0, 1)

# Transformer layer with Rotary Embeddings and RMSNorm
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Attention mechanism
        self.qkv_proj = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.o_proj = nn.Linear(dim_head * heads, dim, bias=False)
        self.rotary_emb = RotaryEmbedding(dim_head)

        # Feed Forward layer
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )

        # Normalization
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, mask):
        B, T, C = x.shape
        
        # Apply RMSNorm before attention
        x = self.norm1(x)

        # Compute Q, K, V
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.heads, -1).permute(0, 2, 1, 3), qkv)

        # Apply rotary embeddings to q and k
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Attention scores
        # scores = torch.einsum('bthd,bThd->bhtT', q, k) * self.scale
        # attn = scores.softmax(dim=-1)
        square_mask = create_square_mask(mask)
        # print(square_mask.unsqueeze(-2).repeat(1, self.heads, 1, 1).size(), q.size())
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=(square_mask==1).unsqueeze(-3).repeat(1, self.heads, 1, 1), dropout_p=0.25).permute(0, 2, 1, 3).flatten(-2, -1)
        # print("isnan: ", out.isnan().any())

        # Apply attention
        # out = torch.einsum('bhtT,bThd->bthd', attn, v)
        # out = out.reshape(B, T, C)
        out = self.o_proj(out)

        # Residual connection and normalization
        x = x + out
        x = x + self.ff(self.norm2(x))
        
        return x

class Pooler(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.lin = nn.Linear(dim, n_classes)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x): 
        # out = self.lin(self.dropout(x[:, 0]))
        out = self.lin(self.dropout(x.mean(dim=-2)))
        # out = F.sigmoid(out)
        # print(out)
        return out

# Transformer model with multiple layers
class Transformer(nn.Module):
    def __init__(self, num_layers, dim, n_classes, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Embedding(128, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, ff_mult, dropout) for _ in range(num_layers)
        ])
        self.pooler = Pooler(dim, n_classes)

    def forward(self, x, att, pool: bool = True):
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x, att)
        return self.pooler(x) if pool else x

