from torch import nn
from .Unet import Unet
from .AutoencoderKL import AutoencoderKL
import open_clip
import torch
class LatentDiffusionInpainting(nn.Module):
    def __init__(self):
        #self.diffusion_model = Unet()
        #self.first_stage_model = AutoencoderKL()
        self.cond_stage_model, _, _ = open_clip.create_model_and_transforms(arch="ViT-H-14", device=torch.device('cpu'), pretrained="laion2b_s32b_b79k")
        del self.cond_stage_model.visual
        self.max_length = 77

    def forward(self, x):
        return
    def get_conditional(self, text):
        tokens = open_clip.tokenize(text)
        x = self.cond_stage_model.token_embedding(tokens)
        x = x + self.cond_stage_model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i, r in enumerate(self.cond_stage_model.transformer.resblocks):
            if i == len(self.cond_stage_model.transformer.resblocks) - 1:
                break
            x = r(x, attn_mask=self.cond_stage_model.attn_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.cond_stage_model.ln_final(x)
        return x


if __name__ == "__main__":
    model = LatentDiffusionInpainting()
    text = "A photo of a pizza"
    cond = model.get_conditional(text)