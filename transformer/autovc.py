import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformer.Constants as Constants


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre=384):
        super(Decoder, self).__init__()
        
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(512, 1024, 3, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        # x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class AutoVC(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck=32, dim_emb=256, dim_pre=384, freq=32):
        super(AutoVC, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.src_pitch_emb = nn.Embedding(100, dim_neck*2, padding_idx=Constants.PAD)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, pitch, c_org, c_trg):
                
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)

        pitch_encoder_output = self.src_pitch_emb(pitch)
        encoder_outputs = torch.cat((code_exp, pitch_encoder_output, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)


def prepocess_autovc(x, pitch, L, crop_len=128):

    lengths = L-1
    
    lengths = lengths.long()  
    lengths = lengths*(lengths>=crop_len) + (lengths<crop_len)*crop_len
    sampling_Length = torch.tensor([crop_len]).long().to(lengths.device)
    mels = []
    pitchs = []

    for mel, pitch, length in zip(x, pitch, lengths):
        offset = torch.randint(
            low= 0,
            high= int(length - sampling_Length + 1),
            size= (1,)
            ).to(x.device)
        mels.append(mel[offset:offset + sampling_Length, :])
        pitchs.append(pitch[offset:offset + sampling_Length])

    mels = torch.stack(mels)
    pitchs = torch.stack(pitchs)

    if L.max()<crop_len:
      pitchs = torch.cat((pitchs, torch.zeros(L.shape[0],int(crop_len-L.max())).int().to(pitchs.device)), dim=1)
      mels = torch.cat((mels, -10*torch.ones(L.shape[0],int(crop_len-L.max()), mels.shape[-1]).float().to(mels.device)), dim=1)

    return mels, pitchs

def synth_autovc(batch, model=None, pitch_shift=True, crop_len=128):

    x = batch['mel']
    pitch = batch['f0p']
    L = batch['length_mel']

    if pitch_shift:
      shiftAB = batch['ref_pc']-batch['pc']
      pitch = ((pitch+shiftAB.unsqueeze(1))*((pitch!=0).int())).detach()

    pad = crop_len - L.max()%crop_len

    if pad!=0:
      pitch = torch.cat((pitch, torch.zeros(L.shape[0],int(pad)).int().to(pitch.device)), dim=1)
      x = torch.cat((x, -10*torch.ones(L.shape[0],int(pad), x.shape[-1]).float().to(x.device)), dim=1)

    results = []
    for i in range(0,x.shape[1],128):
      _, x_identic_psnt, _ = model(x[:,i:i+128], pitch[:,i:i+128], batch['se'], batch['ref_se'])
      results.append(x_identic_psnt)

    result = torch.cat(results, dim=1)
    result = result[:,:L.max(),:]

    return result

  
    