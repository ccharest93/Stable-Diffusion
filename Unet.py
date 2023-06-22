import torch
from torch import nn
from layers.util import conv_nd, GroupNorm32, zero_module
from layers.ResBlock import ResBlock, Downsample, Upsample
from layers.SpatialTransformer import SpatialTransformer
import math
from einops import repeat

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 32
        self.in_channels = 9
        self.model_channels = 320
        self.out_channels = 4
        self.num_res_blocks = [2,2,2,2]
        self.attention_resolutions = [4,2,1]
        self.dropout = 0
        self.channel_mult = [1,2,4,4]
        self.conv_resample = True
        self.num_classes = None
        self.use_checkpoint = True
        self.dtype = torch.float32
        self.num_heads = -1
        self.num_head_channels = 64
        self.num_heads_upsample = -1
        self.predict_codebook_ids = False
        self.dims = 2

        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.model_channels, 3, padding=1)
        ])
        self._feature_size = self.model_channels
        input_block_chans = [self.model_channels]
        ch = self.model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        out_channels= mult * self.model_channels,
                        emb_channels= time_embed_dim
                    )
                ]
                
                ch = mult * self.model_channels
                if ds in self.attention_resolutions:
                    num_heads = ch // self.num_head_channels
                    dim_head = self.num_head_channels
                    layers.append(SpatialTransformer(ch, num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(Downsample(ch, out_channels=out_ch))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        
        self.middle_block = nn.Sequential(
            ResBlock(ch, out_channels=ch, emb_channels=time_embed_dim),
            SpatialTransformer(ch, num_heads),
            ResBlock(ch, out_channels=ch, emb_channels=time_embed_dim),)
        
        self._feature_size += ch       
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, out_channels=self.model_channels * mult,emb_channels=time_embed_dim)
                ]
                ch = self.model_channels * mult
                if ds in self.attention_resolutions:
                    num_heads = ch // self.num_head_channels
                    dim_head = self.num_head_channels
                    layers.append(SpatialTransformer(ch, num_heads))

                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(Upsample(ch, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, self.out_channels, 3, padding=1))
    def forward(self,x, timesteps=None,context=None):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)

        for module in self.input_blocks:
            if isinstance(module, Downsample):
                h = module(h)
            if isinstance(module, nn.Conv2d):
                h = module(h)
            if isinstance(module, nn.ModuleList):
                for timestep in module:
                    if isinstance(timestep, ResBlock):
                        h = timestep(h, emb=emb, context=context)
                    if isinstance(timestep, SpatialTransformer):
                        h = timestep(h, emb=emb, context=context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)