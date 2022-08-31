import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import hparams

def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(torch.from_numpy(x), (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.shape[0] > max_len:
            raise ValueError("not max_len")

        s = x.shape[1]
        x_padded = F.pad(torch.from_numpy(x), (0, 0, 0, max_len-x.shape[0]))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.shape[0] for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output

def pad_log_2D_tensor(inputs, maxlen=None, pad_value=-10.0):

    def pad(x, max_len, pad_value):
        if x.shape[0] > max_len:
            raise ValueError("not max_len")

        s = x.shape[1]
        x_padded = F.pad(torch.from_numpy(x), (0, 0, 0, max_len-x.shape[0]), value=pad_value)
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen, pad_value) for x in inputs])
    else:
        max_len = max(x.shape[0] for x in inputs)
        output = torch.stack([pad(x, max_len, pad_value) for x in inputs])

    return output