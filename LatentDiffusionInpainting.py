from torch import nn
from Unet import Unet
from AutoencoderKL import AutoencoderKL
import open_clip
import torch
import numpy as np
from einops import repeat    
def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch

class LatentDiffusionInpainting(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = Unet()
        self.first_stage_model = AutoencoderKL()
        self.cond_stage_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", device=torch.device('cpu'), pretrained="laion2b_s32b_b79k")
        del self.cond_stage_model.visual
        self.max_length = 77
        self.scale_factor = 0.18215

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

from PIL import Image
if __name__ == "__main__":
    with torch.no_grad():
        input_image = {"image": Image.open("test.jpg"), "mask": Image.open("test.jpg")}
        prompt = "A painting of a cat"
        ddim_steps = 45
        num_samples = 4
        scale = 10
        seed = 100

        #predict function
        init_image = input_image["image"].convert("RGB")
        init_mask = input_image["mask"].convert("RGB")
        image = pad_image(init_image) # resize to integer multiple of 32
        mask = pad_image(init_mask) # resize to integer multiple of 32
        w, h = image.size
        print("Inpainting...", w, h)

        #inpaint function
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = LatentDiffusionInpainting()
        prng = np.random.RandomState(seed) #rng
        start_code = prng.randn(num_samples, 4, h // 8, w // 8) #random start code
        start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32) #to device
        batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)
        c = model.get_conditional(batch["txt"])

        c_cat = list()
        for ck in ['mask', 'masked_image']:
            cc = batch[ck].float()
            if ck != 'masked_image':
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.scale_factor * model.first_stage_model.encode(cc).sample()
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)





