from torch import nn
import torch
from layers.ResBlock import ResBlock, Downsample, Upsample
from layers.SpatialTransformer import MemoryEfficientAttnBlock
import numpy as np

#Sampling distribution for prior
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch_mult = [1, 2, 4, 4] #mult for channel at each resolution of encoding
        self.num_resolutions = len(ch_mult) #nb of resolutions in encoder
        self.num_res_blocks = 2  #residual block at each resolution
        in_channels = 3
        emb_channels = 128
        time_emb_channels = 0
        out_channels = 4  

        block_in = emb_channels #channels at starting resolution

        self.conv_in = torch.nn.Conv2d(in_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = emb_channels*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         time_emb_channels=time_emb_channels,
                                         ))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, padding=(0,1,0,1))
            self.down.append(down)
        
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       time_emb_channels=time_emb_channels)
        self.mid.attn_1 = MemoryEfficientAttnBlock(block_in)
        self.mid.block_2 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       time_emb_channels=time_emb_channels)
        
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    
    def forward(self, x):
        #encoding
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], None)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, t_emb=None, context=None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, t_emb=None, context=None)

        # end
        h = self.norm_out(h)
        h = nn.SiLU()(h)
        h = self.conv_out(h)
        return h    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch_mult = [1, 2, 4, 4] #mult for channel at each resolution of decoding
        self.num_resolutions = len(ch_mult) #nb of resolutions in decoder
        self.num_res_blocks = 2 #residual block at each resolution
        in_channels = 4
        emb_channels = 128
        time_emb_channels = 0 # needed for timeblock diffusion
        out_channels = 3

        block_in = emb_channels*ch_mult[self.num_resolutions-1] #channels at starting resolution

        # in_channels to block_in
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       time_emb_channels=time_emb_channels,
                                       )
        self.mid.attn_1 = MemoryEfficientAttnBlock(block_in)
        self.mid.block_2 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       time_emb_channels=time_emb_channels,
                                       )

        #decoding
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = emb_channels*ch_mult[i_level]
            for _ in range(self.num_res_blocks+1):
                block.append(ResBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         time_emb_channels=time_emb_channels,))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, t_emb=None, context=None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, t_emb=None, context=None)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, t_emb = None, context = None)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nn.SiLU()(h)
        h = self.conv_out(h)
        return h
    
class AutoencoderKL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0)
        self.post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec