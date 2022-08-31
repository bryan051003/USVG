import torch
import torch.nn as nn
import numpy as np
import hparams as hp

import transformer.Constants as Constants
from transformer.Layers import FFTBlock, PreNet, PostNet, Linear


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 n_src_vocab=hp.vocab_size,
                 len_max_seq=hp.vocab_size,
                 d_word_vec=hp.encoder_dim,
                 n_layers=hp.encoder_n_layer,
                 n_head=hp.encoder_head,
                 d_k=hp.encoder_dim // hp.encoder_head,
                 d_v=hp.encoder_dim // hp.encoder_head,
                 d_model=hp.encoder_dim,
                 d_inner=hp.encoder_conv1d_filter_size,
                 dropout=hp.dropout):

        super(Encoder, self).__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab,
                                         d_word_vec,
                                         padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout, phon=True) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 len_max_seq=hp.max_seq_len,
                 n_layers=hp.decoder_n_layer,
                 n_head=hp.decoder_head,
                 d_k=hp.decoder_dim // hp.decoder_head,
                 d_v=hp.decoder_dim // hp.decoder_head,
                 d_model=hp.decoder_dim,
                 d_inner=hp.decoder_conv1d_filter_size,
                 dropout=hp.dropout,
                 Position=True, 
                 query_projection=False):

        super(Decoder, self).__init__()

        n_position = len_max_seq + 1

        self.Position = Position
        self.query_projection = query_projection

        if Position:
          self.position_enc = nn.Embedding.from_pretrained(
              get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
              freeze=True)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout, query_projection=query_projection) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False, hidden_query=None):

        if self.query_projection:
            assert hidden_query is not None, "Query should be given for the Excitation Generator."

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        if self.Position:
          dec_output = enc_seq + self.position_enc(enc_pos)
        else:
          dec_output = enc_seq

        for i, dec_layer in enumerate(self.layer_stack):
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask, hidden_query=hidden_query if i==0 else None)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


####

import math 

def maximum_path_numpy(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path

class MLE_Loss(torch.nn.modules.loss._Loss):
    def forward(self, z, mean, std):
        loss = torch.sum(std) + 0.5 * torch.sum(torch.exp(-2 * std) * (z - mean) ** 2) 
        loss /= z.shape[2]*z.shape[1]

        return loss

class Phon2Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 decoder=True,
                 enc_layers=3,
                 dec_layers=2,
                 d_model=128,
                 d_inner=512,
                 noise_weight=0.1):

        super(Phon2Encoder, self).__init__()

        self.phon_encoder = Encoder(d_word_vec=d_model,
                                    n_layers=enc_layers,
                                    d_k=d_model // hp.encoder_head,
                                    d_v=d_model // hp.encoder_head,
                                    d_model=d_model,
                                    d_inner=d_inner)
        if decoder:
          self.phon_decoder = Decoder(len_max_seq=1500,
                                      n_layers=dec_layers,
                                      d_k=d_model // hp.encoder_head,
                                      d_v=d_model // hp.encoder_head,
                                      d_model=d_model,
                                      d_inner=d_inner)
        else:
          self.phon_decoder = None

        #self.duration_predictor = Duration_Predictor(d_model, nphon)
        self.Project = torch.nn.Conv1d( in_channels= d_model, out_channels= d_model * 2, kernel_size= 1  )
        self.d_model = d_model
        
        self.MLE = MLE_Loss()
        self.mse = nn.MSELoss()
        self.noise_weight = noise_weight

    def LR(self, x, duration, spec_max_length=None): # from length regulator
        with torch.no_grad():
          expand_max_len = torch.max(
              torch.sum(duration, -1), -1)[0]
          alignment = torch.zeros(duration.size(0), expand_max_len, duration.size(1)).numpy()
          alignment = create_alignment(alignment, duration.cpu().numpy())
          alignment = torch.from_numpy(alignment).detach().to(device)

        output = alignment @ x
        if spec_max_length:
            output = F.pad(
                output, (0, 0, 0, spec_max_length-output.size(1), 0, 0))
        return output

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, batch, z=None, encodeB=False, dur=False, sg=False):

        if encodeB:
          phon = batch["phonB"]
          spec_pos = batch['refB_pos']
          phon_pos = batch["phonB_pos"]
          spec_max_length = batch["refB_spec_max_len"]

        else:
          phon = batch["phon"]
          spec_pos = batch["spec_pos"]
          phon_pos = batch["phon_pos"]
          spec_max_length = batch["spec_max_len"]
          mus_mask = (batch['singer']<124)[:, None, None]

        phon_output, _ = self.phon_encoder(phon, phon_pos)
        lengths = ((phon_pos>0).int().sum(dim=1))
        mask = ((torch.arange(torch.max(lengths))[None, :].to(lengths.device) < lengths[:, None] ).float()).unsqueeze(-1) # new mask
         
        phon_encoder_output = self.Project(phon_output.transpose(1, 2)) # [b,d*2,t]
        phon_encoder_output = phon_encoder_output.transpose(1, 2) * mask
        mean, log_Std = torch.split( phon_encoder_output.transpose(1, 2), [self.d_model, self.d_model], dim= 1 )
        
        y_mask = get_non_pad_mask(spec_pos)
        attn_mask = torch.unsqueeze(mask.transpose(1,2), -1) * torch.unsqueeze(y_mask.transpose(1,2), 2)
        
        z = z.transpose(1, 2)

        with torch.no_grad():
            o_scale = torch.exp(-2 * log_Std)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - log_Std, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, T] = [b, t, T]
            logp3 = torch.matmul((mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, T] = [b, t, T]
            logp4 = torch.sum(-0.5 * (mean ** 2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, T]
            attn = maximum_path_numpy(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        mel_Mean = torch.matmul(attn.squeeze(1).transpose(1, 2), mean.transpose(1, 2)).transpose(1, 2)  # [b, T, t], [b, t, d] -> [b, d, T]
        mel_Log_Std = torch.matmul(attn.squeeze(1).transpose(1, 2), log_Std.transpose(1, 2)).transpose(1, 2)  # [b, T, t], [b, t, d] -> [b, d, T]

        duration = torch.sum(attn, -1) * mask
        if dur:
          return duration

        mleloss = self.MLE(z,mel_Mean*mus_mask,mel_Log_Std*mus_mask)

        if self.phon_decoder!=None:
          phon_output = torch.matmul(attn.squeeze(1).transpose(1, 2), phon_output)  # [b, T, t], [b, t, d] -> [b, T, d]
          phon_output = self.mask_tensor(phon_output, spec_pos, spec_max_length)
          if sg:
            phon_output = phon_output.detach()
          phon_output = self.phon_decoder(phon_output, spec_pos)
          phon_output = self.mask_tensor(phon_output, spec_pos, spec_max_length)
          eloss = self.mse(phon_output*mus_mask, z.transpose(1, 2))

        else:
          noises = torch.randn_like(mel_Mean)
          phon_output = (mel_Mean + torch.exp(mel_Log_Std) * noises)
          phon_output = self.mask_tensor(phon_output.transpose(1, 2), spec_pos, spec_max_length)
          eloss = torch.zeros(1).to(phon_output.device)

        return phon_output, mleloss, eloss
        

class MelEncoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 d_model=128,
                 n_layers=2,
                 dropout=hp.dropout,
                 n_speaker=124,
                 adv_weight=0,
                 query_projection=False,
                 d_hid=128):

        super(MelEncoder, self).__init__()
        self.prenet = nn.Sequential(
            nn.Conv1d(hp.num_mels, d_model, kernel_size=hp.fft_conv1d_kernel[0], padding=hp.fft_conv1d_padding[0]),
            nn.InstanceNorm1d(d_model, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer_stack = nn.ModuleList([PositionwiseFeedForward(
            d_model, d_hid, dropout=dropout, use_LN=True, old_ver=False) for _ in range(n_layers)])
        
        self.adv_weight = adv_weight
        if self.adv_weight:
          self.speaker_classifier = speaker_classifier(d_model=d_model, n_logits=n_speaker, adv_weight=self.adv_weight)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, batch, ref=False, add_noise=False, return_layer_output=False):

        if ref:
          mel = batch["ref_mel"]
          spec_pos = batch["ref_pos"]
          spec_max_length = batch["ref_spec_max_len"]

        else:
          mel = batch["mel"]
          spec_pos = batch["spec_pos"]
          spec_max_length = batch["spec_max_len"]

        if add_noise:
          mel = mel + add_noise*torch.randn(mel.size()).to(mel.device)

        output = mel.transpose(1, 2)
        output = self.prenet(output)
        output = output.transpose(1, 2)

        if return_layer_output:
          layer_out = output

        for i, enc_layer in enumerate(self.layer_stack):
            output = enc_layer(output)
            if return_layer_output:
              layer_out = torch.cat((layer_out, output), dim=-1)

        output = self.mask_tensor(output, spec_pos, spec_max_length)
        
        floss = torch.zeros(1).to(mel.device)
        sloss = torch.zeros(1).to(mel.device)

        if self.adv_weight:
          sloss = self.speaker_classifier(output, batch['singer'], spec_pos)
          # floss = self.f0_classifier(output, batch['f0p'], spec_pos)
        # else:
        #   floss = torch.zeros(1).to(mel.device)
        #   sloss = torch.zeros(1).to(mel.device)

        if return_layer_output:
          layer_out = self.mask_tensor(layer_out, spec_pos, spec_max_length)
          return output, layer_out, sloss, floss
        else:
          return output, sloss, floss

    def inference(self, mel, spec_pos, spec_max_length, return_layer_output=False):

        output = mel.transpose(1, 2)
        output = self.prenet(output)
        output = output.transpose(1, 2)

        if return_layer_output:
          layer_out = output

        for i, enc_layer in enumerate(self.layer_stack):
            output = enc_layer(output)
            if return_layer_output:
              layer_out = torch.cat((layer_out, output), dim=-1)

        output = self.mask_tensor(output, spec_pos, spec_max_length)

        if return_layer_output:
          layer_out = self.mask_tensor(layer_out, spec_pos, spec_max_length)
          return output, layer_out
        else:
          return output