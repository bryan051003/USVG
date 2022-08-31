import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn



class CosinSimAttention(nn.Module):

    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        q_norm = torch.norm(q, p=2, dim=2, keepdim=True).clamp(min=1e-12)
        q_norm = torch.div(q, q_norm)
        k_norm = torch.norm(k, p=2, dim=2, keepdim=True).clamp(min=1e-12)
        k_norm = torch.div(k, k_norm)
        attn = torch.bmm(q_norm, k_norm.transpose(1, 2))+1

        if mask is not None:
            attn = attn.masked_fill(mask, 0.0)

        attn = attn/(attn.sum(dim=2, keepdim=True).clamp(min=1e-12))
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
