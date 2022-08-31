import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer.Modules import ScaledDotProductAttention
import hparams as hp


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, use_LN=True, old_ver=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        
        self.use_LN = use_LN
        self.layer_norm = nn.LayerNorm(d_model) if use_LN or old_ver else None

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if self.use_LN:
            output = self.layer_norm(output + residual)
        else:
            output = output + residual

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, use_LN=True, old_ver=False, phon=False):
        super().__init__()

        # Use Conv1D
        if phon:
          # position-wise
          self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=3, padding=1)
        else:
          # position-wise
          self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=hp.fft_conv1d_kernel[0], padding=hp.fft_conv1d_padding[0])
          
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=hp.fft_conv1d_kernel[1], padding=hp.fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in) if use_LN or old_ver else None
        self.use_LN = use_LN
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        if self.use_LN:
            output = self.layer_norm(output + residual)
        else:
            output = output + residual

        return output


class FeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, use_LN=True, old_ver=False):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=hp.fft_conv1d_kernel[1], padding=hp.fft_conv1d_padding[1])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=hp.fft_conv1d_kernel[1], padding=hp.fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in) if use_LN or old_ver else None
        self.use_LN = use_LN
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        if self.use_LN:
            output = self.layer_norm(output + residual)
        else:
            output = output + residual

        return output