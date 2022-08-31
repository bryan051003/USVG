import torch
from torch import nn
import torch.nn.functional as F
from transformer.Layers import Linear
import hparams as hp

class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, 
                 input_size, 
                 output_size, 
                 gain=2 ** 0.5, 
                 use_wscale=False, 
                 lrmul=1, 
                 bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        output = F.linear(x, self.weight * self.w_mul, bias)
        return output

class StyleMod(nn.Module):
    def __init__(self, 
                 latent_size=hp.singer_latent_size, 
                 channels=hp.encoder_dim, 
                 use_wscale=True):
        super(StyleMod, self).__init__()
        # self.PreNet = nn.Sequential(
        #     Linear(latent_size, latent_size),
        #     nn.ReLU(),
        #     nn.Dropout(hp.dropout)
        # )
        self.lin = EqualizedLinear(latent_size,
                                   channels * 2,
                                   gain=1.0, 
                                   use_wscale=use_wscale)

    
    def AdaIn(self, feat, mask, style_mean, style_std):
        size = feat.size()

        mask = (mask > 0).float().unsqueeze(-1)
        mask_ = mask.expand(size)
        feat_mean = (feat * mask_).sum(dim=1) / mask_.sum(dim=1)
        feat_mean = feat_mean.unsqueeze(1).expand(size)
        deviations = (feat - feat_mean) ** 2
        deviations = deviations * mask_
        # Variance
        feat_var = deviations.sum(dim=1) / (mask_.sum(dim=1)-1)

        feat_std = feat_var.sqrt().unsqueeze(1).expand(size)

        #AdaIn
        normalized_feat = (feat - feat_mean) / (feat_std+1e-8)
        output_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
        return output_feat * mask_
    
    def AdaLN(self, feat, mask, style_mean, style_std):
        size = feat.size()

        mask = (mask > 0).float().unsqueeze(-1)
        mask_ = mask.expand(size)
        mu  = torch.mean(feat, dim=-1, keepdim=True)
        sigma = torch.std(feat, dim=-1, keepdim=True)
        normalized_feat = (feat - mu) / (sigma+1e-8) # [B, T, H_m]
        output_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
        return output_feat * mask_
    
    def forward(self, x, latent, mask, IN=True):
        # latent = self.PreNet(latent)
        style = self.lin(latent)  # [batch_size, 1, latent_size] => [batch_size, 1, n_channels*2]
        

        shape = [-1, 1, 2, x.size(-1)]
        style = style.view(shape)  # [batch_size, 1, 2, n_channels]
        
        style_std = style[:,:,0] +1
        style_mean = style[:,:,1]
        if IN:
          x = self.AdaIn(x, mask, style_mean, style_std)
        else:
          x = self.AdaLN(x, mask, style_mean, style_std)
        return x


class StyleMod2d(nn.Module):
    def __init__(self, 
                 latent_size=hp.singer_latent_size, 
                 channels=hp.encoder_dim, 
                 PN=True,
                 IN=False,
                 use_wscale=True):
        super(StyleMod2d, self).__init__()

        self.IN = IN

        if PN:
          self.PreNet = nn.Sequential(
              Linear(latent_size, latent_size),
              nn.ReLU(),
              nn.Dropout(hp.dropout)
          )
        else:
          self.PreNet = False
        self.lin = EqualizedLinear(latent_size,
                                   channels * 2,
                                   gain=1.0, 
                                   use_wscale=use_wscale)
    
    def AdaLN(self, feat, mask, style_mean, style_std):
        size = feat.size()

        mask = (mask > 0).float().unsqueeze(-1)
        mask_ = mask.expand(size)
        mu  = torch.mean(feat, dim=-1, keepdim=True)
        sigma = torch.std(feat, dim=-1, keepdim=True)
        normalized_feat = (feat - mu) / (sigma+1e-8) # [B, T, H_m]
        output_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
        return output_feat * mask_

    def AdaIN(self, feat, mask, style_mean, style_std):
        size = feat.size()

        mask = (mask > 0).float().unsqueeze(-1)
        mask_ = mask.expand(size)
        feat_mean = (feat * mask_).sum(dim=1) / mask_.sum(dim=1)
        feat_mean = feat_mean.unsqueeze(1).expand(size)
        deviations = (feat - feat_mean) ** 2
        deviations = deviations * mask_
        # Variance
        feat_var = deviations.sum(dim=1) / (mask_.sum(dim=1)-1)

        feat_std = feat_var.sqrt().unsqueeze(1).expand(size)

        #AdaIn
        normalized_feat = (feat - feat_mean) / (feat_std+1e-8)
        output_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
        return output_feat * mask_
    
    def forward(self, x, latent, mask, PN=True):
        if self.PreNet:
          latent = self.PreNet(latent)
        style = self.lin(latent)  # [batch_size, time, latent_size] => [batch_size, time, n_channels*2]
        

        shape = [-1, x.size(1), 2, x.size(-1)]
        style = style.view(shape)  # [batch_size, time, 2, n_channels]
        
        style_std = style[:,:,0] +1
        style_mean = style[:,:,1]
        if self.IN:
          x = self.AdaIN(x, mask, style_mean, style_std)
        else:
          x = self.AdaLN(x, mask, style_mean, style_std)
        return x