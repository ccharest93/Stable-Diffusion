from torch import nn
import torch

class Upsample(nn.Module):
    def __init__(self,
                  in_channels,
                  out_channels=None,
                  padding=(1,1,1,1)):
        super().__init__()
        out_channels = out_channels or in_channels
        #padding
        self.padding = padding
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = torch.nn.functional.pad(x, self.padding, mode="constant", value=0)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels=None, 
                 padding=(1,1,1,1)):
        super().__init__()
        self.padding = padding
        out_channels = out_channels or in_channels
        self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2)

    def forward(self, x):
        x = torch.nn.functional.pad(x, self.padding, mode="constant", value=0)
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
    ):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if emb_channels > 0:
            self.emb_proj = nn.Linear(emb_channels,out_channels),
        else:
            self.emb_proj = nn.Identity()

        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d( in_channels, out_channels, 1)

    def forward(self, x, emb):
        h = self.norm1(x)
        h = nn.SiLU()(h)
        h = self.conv1(h)

        if emb is not None:
            h = h + self.emb_proj(nn.SiLU()(emb))[:,:,None,None]
        
        h = self.norm2(h)
        h = nn.SiLU()(h)
        h = self.conv2(h)

        return self.skip_connection(x) + h
        return self.skip_connection(x) + h, (x, emb), self.parameters()
