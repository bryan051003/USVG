"""Convolutional transsformer"""

from typing import Optional, Tuple

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Dropout, LayerNorm, Conv1d, MultiheadAttention

from transformer.StyleMod import StyleMod2d


class Smoother(Module):
    """Convolutional Transformer Encoder Layer"""

    def __init__(self, d_model: int, nhead: int, d_hid: int, dropout=0.1):
        super(Smoother, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.conv1 = Conv1d(d_model, d_hid, 9, padding=4)
        self.conv2 = Conv1d(d_hid, d_model, 1, padding=0)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # multi-head self attention
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]

        # add & norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # conv1d
        src2 = src.transpose(0, 1).transpose(1, 2)
        src2 = self.conv2(F.relu(self.conv1(src2)))
        src2 = src2.transpose(1, 2).transpose(0, 1)

        # add & norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Extractor(Module):
    """Convolutional Transformer Decoder Layer"""

    def __init__(
        self, d_model: int, nhead: int, d_hid: int, d_k=9, dropout=0.1, no_residual=False, SALN=False,
    ):
        super(Extractor, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.conv1 = Conv1d(d_model, d_hid, d_k, padding=d_k//2)
        self.conv2 = Conv1d(d_hid, d_model, 1, padding=0)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.no_residual = no_residual
        self.SALN = SALN
        if self.SALN:
          self.StyleMod = StyleMod2d(latent_size=d_model, channels=d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # multi-head self attention
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # multi-head cross attention
        tgt2, attn = self.cross_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        # tgt: (tgt_len, batch, d_model)
        if self.SALN:
            tgt2 = self.norm2(tgt2)
            tgt = self.StyleMod(tgt.transpose(0, 1), tgt2.transpose(0, 1), 1-tgt_key_padding_mask.float())
            tgt = tgt.transpose(0, 1)

        else:
            # add & norm
            if self.no_residual:
                tgt = self.dropout2(tgt2)
            else:
                tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # conv1d
        tgt2 = tgt.transpose(0, 1).transpose(1, 2)
        tgt2 = self.conv2(F.relu(self.conv1(tgt2)))
        tgt2 = tgt2.transpose(1, 2).transpose(0, 1)

        # add & norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn
