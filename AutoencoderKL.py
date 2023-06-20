from torch import nn
import torch
from layers.ResBlock import ResBlock, Downsample, Upsample
from layers.SpatialTransformer import MemoryEfficientAttnBlock
import numpy as np
class Encoder(nn.Module):
    def __init__(self, 
                 *, 
                 ch, 
                 out_ch, 
                 ch_mult=(1,2,4,8), 
                 num_res_blocks,
                 attn_resolutions, 
                 dropout=0.0, 
                 resamp_with_conv=True, 
                 in_channels,
                 resolution, 
                 z_channels, 
                 double_z=True, 
                 use_linear_attn=False, 
                 attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        ch = 128
        out_ch = 3
        ch_mult = [1, 2, 4, 4]
        num_res_blocks = 2
        attn_resolutions = []
        dropout = 0.0
        resamp_with_conv = True
        in_channels = 3
        resolution = 256
        z_channels = 4
        double_z = True
        use_linear_attn = False
        attn_type = "vanilla"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         emb_channels=self.temb_ch,
                                         ))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, padding=(0,1,0,1))
                curr_res = curr_res // 2
            self.down.append(down)
        
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=self.temb_ch)
        self.mid.attn_1 = MemoryEfficientAttnBlock(block_in)
        self.mid.block_2 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=self.temb_ch)
        
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

class Decoder(nn.Module):
    def __init__(self, 
                 *, 
                 ch, 
                 out_ch, 
                 ch_mult=(1,2,4,8),
                   num_res_blocks,
                 attn_resolutions, 
                 dropout=0.0, 
                 resamp_with_conv=True, 
                 in_channels,
                 resolution, 
                 z_channels, 
                 give_pre_end=False, 
                 tanh_out=False, 
                 use_linear_attn=False,
                 attn_type="vanilla",
                   **ignorekwargs):
        super().__init__()
        self.ch = 128
        self.out_ch = 3
        self.ch_mult = [1, 2, 4, 4]
        self.num_res_blocks = 2
        self.attn_resolutions = []
        self.dropout = 0.0
        self.resamp_with_conv = True
        self.in_channels = 3
        self.resolution = 256
        self.z_channels = 4
        self.give_pre_end = False
        self.tanh_out = False
        self.use_linear_attn = False

        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=self.temb_ch,
                                       )
        self.mid.attn_1 = MemoryEfficientAttnBlock(block_in)
        self.mid.block_2 = ResBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       emb_channels=self.temb_ch,
                                       )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks+1):
                block.append(ResBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         emb_channels=self.temb_ch,))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

class AutoencoderKL(nn.Module):
    def __init__(self):
        super().__init__()
        dd_config = {'double_z': True,
                      'z_channels': 4, 
                      'resolution': 256, 
                      'in_channels': 3, 
                      'out_ch': 3, 
                      'ch': 128, 
                      'ch_mult': [1, 2, 4, 4], 
                      'num_res_blocks': 2, 
                      'attn_resolutions': [], 
                      'dropout': 0.0}
        lossconfig = {'target': 'torch.nn.Identity'}
        embed_dim = 4
        ckpt_path = ""
        ignore_keys =[]
        image_key = 'image'
        colorize_nlabels = None
        monitor = 'val/rec_loss'
        ema_decay = None
        learn_logvar = False

        self.encoder = Encoder(**dd_config)
        self.decoder = Decoder(**dd_config)
        self.loss = torch.nn.Identity()
        self.quant_conv = torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0)
        self.post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)
        self.embed_dim = embed_dim
        self.monitor = monitor

