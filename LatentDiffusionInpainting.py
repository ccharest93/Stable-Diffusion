from torch import nn
from .Unet import Unet
class LatentDiffusionInpainting(nn.Module):
    def __init__(self):
        self.diffusion_model = Unet()
