from torch import nn
import torch
from layers.ResBlock import ResBlock, Downsample
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
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, padding=(0,1,0,1))
                curr_res = curr_res // 2
            self.down.append(down)


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

