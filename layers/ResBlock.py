from torch import nn
from torch.nn import functional as F
from layers.util import conv_nd, zero_module, GroupNorm32

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, 
                 channels, 
                 dims=2, 
                 out_channels=None, 
                 padding=1):
        super().__init__()
        self.dims = dims
        out_channels = out_channels or channels

        self.conv = conv_nd(dims, channels, out_channels, 3, padding=padding)

    def forward(self, x):
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, 
                 channels, 
                 dims=2, 
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.dims = dims
        out_channels = out_channels or channels
        stride = 2 if dims != 3 else (1, 2, 2)

        self.op = conv_nd( dims, channels, out_channels, 3, stride=stride, padding=padding)

    def forward(self, x):
        return self.op(x)

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
        channels,
        out_channels
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = 1280
        self.dropout = 0
        self.out_channels = out_channels
        self.use_conv = False
        self.use_checkpoint = False
        self.use_scale_shift_norm = False
        self.dims = 2
        self.down = False
        self.up = False

        self.in_layers = nn.Sequential(
            GroupNorm32(self.channels),
            nn.SiLU(),
            conv_nd(self.dims, self.channels, self.out_channels, 3, padding=1),
        )

        self.updown = self.up  or self.down

        self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels,self.out_channels),
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            zero_module(conv_nd(self.dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == self.channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(self.dims, self.channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h, (x, emb), self.parameters()
