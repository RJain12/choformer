import torch
from torch import nn, Tensor
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size//2, num_layers, bidirectional=True) # bidirectional?
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, 1),
        )
                
    def forward(self, x: Tensor) -> Tensor:
        b, l, d = x.size()
        
        # cls_tok_expand = self.cls_token.unsqueeze(0).repeat(b, 1).unsqueeze(1)
        # x = torch.concat([cls_tok_expand, x, cls_tok_expand], dim=-2)
        
        logit = self.lstm(self.in_proj(x))[0].mean(dim=-2) # NOTE: Idea for pooling: add cls token at beginning and end and add - both final state representations, NVM AVERAGE POOLING!!
        return F.sigmoid(self.classifier(logit)) # sigmoid here for mse comparison
