import math
import torch
from torch import nn
from layers.ResBlock import ResBlock, Downsample, Upsample
from layers.SpatialTransformer import SpatialTransformer

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding

#need to clarify the structure from init parameters
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 9
        self.model_channels = 320
        out_channels = 4
        num_res_blocks = [2,2,2,2]
        attention_resolutions = [4,2,1]
        channel_mult = [1,2,4,4]
        num_heads = -1
        num_head_channels = 64

        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, self.model_channels, 3, padding=1)
        ])
        input_block_chans = [self.model_channels]
        ch = self.model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        out_channels= mult * self.model_channels,
                        time_emb_channels= time_embed_dim
                    )
                ]
                
                ch = mult * self.model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    layers.append(SpatialTransformer(ch, num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(Downsample(ch, out_channels=out_ch))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
        
        self.middle_block = nn.ModuleList([
            ResBlock(ch, out_channels=ch, time_emb_channels=time_embed_dim),
            SpatialTransformer(ch, num_heads),
            ResBlock(ch, out_channels=ch, time_emb_channels=time_embed_dim),])
            
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, out_channels=self.model_channels * mult,time_emb_channels=time_embed_dim)
                ]
                ch = self.model_channels * mult
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    layers.append(SpatialTransformer(ch, num_heads))

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(Upsample(ch, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))

        self.out = nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, out_channels, 3, padding=1))
    def forward(self,x, timesteps=None,context=None):
        h = x
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
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

        for module in self.middle_block:
            h = module(h, emb=emb, context=context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, nn.Conv2d):
                h = module(h)
            elif isinstance(module, nn.ModuleList):
                for timestep in module:
                    if isinstance(timestep, ResBlock):
                        h = timestep(h, emb=emb, context=context)
                    if isinstance(timestep, SpatialTransformer):
                        h = timestep(h, emb=emb, context=context)
                    if isinstance(timestep, Upsample):
                        h = timestep(h)
            else:
                pass
        return self.out(h)