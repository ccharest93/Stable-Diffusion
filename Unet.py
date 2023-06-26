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

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 9
        self.emb_channels = 320
        out_channels = 4
        num_res_blocks = [2,2,2,2]
        attention_resolutions = [4,2,1]
        channel_mult = [1,2,4,4]
        num_heads = -1
        num_head_channels = 64

        # Time embedding
        time_embed_dim = self.emb_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.emb_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        #Input embedding
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, self.emb_channels, 3, padding=1)
        ])

        #UNET variables
        input_block_chans = [self.emb_channels]
        in_block = self.emb_channels
        resolution = 1
        #UNET down
        for level, mult in enumerate(channel_mult):
            out_block = mult * self.emb_channels 
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        in_block,
                        out_channels= out_block,
                        time_emb_channels= time_embed_dim
                    )
                ]
                
                in_block = out_block
                if resolution in attention_resolutions:
                    num_heads = in_block // num_head_channels
                    layers.append(SpatialTransformer(in_block, num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(in_block)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(in_block, out_channels=out_block))
                in_block = out_block
                input_block_chans.append(in_block)
                resolution *= 2
        #UNET middle
        self.middle_block = nn.ModuleList([
            ResBlock(in_block, out_channels=out_block, time_emb_channels=time_embed_dim),
            SpatialTransformer(in_block, num_heads),
            ResBlock(in_block, out_channels=out_block, time_emb_channels=time_embed_dim),])
        #UNET up with residual connections
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_block = mult * self.emb_channels 
            for i in range(num_res_blocks[level] + 1):
                res_channels = input_block_chans.pop()
                layers = [
                    ResBlock(in_block + res_channels, out_channels=out_block,time_emb_channels=time_embed_dim)
                ]
                in_block = out_block
                if resolution in attention_resolutions:
                    num_heads = in_block // num_head_channels
                    layers.append(SpatialTransformer(in_block, num_heads))

                if level and i == num_res_blocks[level]:
                    layers.append(Upsample(in_block, out_channels=out_block))
                    resolution //= 2
                self.output_blocks.append(nn.ModuleList(layers))

        #Output embedding
        self.out = nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=in_block),
            nn.SiLU(),
            nn.Conv2d(self.emb_channels, out_channels, 3, padding=1))
    
    def forward(self,x, timesteps=None,context=None):
        h = x
        hs = []
        t_emb = timestep_embedding(timesteps, self.emb_channels)
        del timesteps
        t_emb = self.time_embed(t_emb)
        for module in self.input_blocks:
            if isinstance(module, Downsample):
                h = module(h)
            if isinstance(module, nn.Conv2d):
                h = module(h)
            if isinstance(module, nn.ModuleList):
                for timestep in module:
                    if isinstance(timestep, ResBlock):
                        h = timestep(h, t_emb=t_emb, context=context)
                    if isinstance(timestep, SpatialTransformer):
                        h = timestep(h, t_emb=t_emb, context=context)
            hs.append(h)

        for module in self.middle_block:
            h = module(h, t_emb=t_emb, context=context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, nn.Conv2d):
                h = module(h)
            elif isinstance(module, nn.ModuleList):
                for timestep in module:
                    if isinstance(timestep, ResBlock):
                        h = timestep(h, t_emb=t_emb, context=context)
                    if isinstance(timestep, SpatialTransformer):
                        h = timestep(h, t_emb=t_emb, context=context)
                    if isinstance(timestep, Upsample):
                        h = timestep(h)
            else:
                pass
        
        return self.out(h)