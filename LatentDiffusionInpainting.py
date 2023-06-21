from torch import nn
from Unet import Unet
from AutoencoderKL import AutoencoderKL
import open_clip
import torch
import numpy as np
from einops import repeat    
from functools import partial
from tqdm import tqdm
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

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
class LatentDiffusionInpainting(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.parameterization = 'eps' 
        self.image_size = 64
        self.channels = 4
        self.diffusion_model = Unet()
        #diffusion params
        self.dpm_num_timesteps = 1000
        self.diffusion_param_schedule()
        timesteps, = self.betas.shape
        self.num_timesteps = int(timesteps)
        self.scale_factor = 0.18215
        #submodels init
        self.first_stage_model = AutoencoderKL()
        self.cond_stage_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", device=torch.device('cpu'), pretrained="laion2b_s32b_b79k")
        del self.cond_stage_model.visual
        self.max_length = 77
        #---------- init latent diffusion end
        self.concat_keys = ('mask', 'masked_image')
        self.masked_image_key = 'masked_image'
    def diffusion_param_schedule(self):
        v_posterior = 0.0
        linear_start=1e-4
        linear_end=2e-2
        betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, 1000, dtype=torch.float64) ** 2).numpy()
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_torch(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
    
    def sample(self,
               S,
               shape,
               cond,
               eta= 1.0,
               temperature=1.0,
               x_T=None,
               log_every_t=100, 
               unconditional_guidance_scale=1.0, 
               unconditional_conditioning=None):
        
        #get diffusion timesteps
        num_ddpm_timesteps = S
        c = num_ddpm_timesteps // self.dpm_num_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        ddim_timesteps = ddim_timesteps + 1

        # calculations for variance schedule
        alphacums = self.alphas_cumprod.cpu()
        self.ddim_alphas = alphacums[ddim_timesteps]
        self.ddim_alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - self.ddim_alphas)
        self.ddim_sigmas = eta * np.sqrt((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev))
        
        b = shape[0]
        img = x_T
        timesteps = ddim_timesteps
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            outs = self.p_sample_ddim(img, cond, ts, 
                                    index=index, 
                                    temperature=temperature,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,)
            img, pred_x0 = outs
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        #decode into an image
        x_samples_ddim = model.decode_first_stage(img)
        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,min=0.0, max=1.0)
        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
        return [Image.fromarray(img.astype(np.uint8)) for img in result]

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, temperature=1.,unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = dict()
        #batch the conditional and unconditional embeddings together (use?_)
        for k in c:
            if isinstance(c[k], list):
                c_in[k] = [torch.cat([
                    unconditional_conditioning[k][i],
                    c[k][i]]) for i in range(len(c[k]))]
            else:
                c_in[k] = torch.cat([
                        unconditional_conditioning[k],
                        c[k]])
        xc = torch.cat([x_in] + c_in['c_concat'], dim=1)
        cc = torch.cat(c_in['c_crossattn'], 1)
        out = self.diffusion_model(xc, t_in, context=cc)
        model_uncond, model_t = out.chunk(2)
        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        e_t = model_output

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas =  self.ddim_sqrt_one_minus_alphas
        sigmas =  self.ddim_sigmas
        
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)

        # current prediction for x_0
        pred_x0 =  extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * x - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * model_output

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, False) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
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
                model.first_stage_model =model.first_stage_model.to(device)
                cc = model.scale_factor * model.first_stage_model.encode(cc).sample()
                model.first_stage_model = model.first_stage_model.to(torch.device('cpu'))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        #retrieving the conditioning for a null_label
        uc_cross = model.get_conditional("")
        uc_cross = repeat(uc_cross, '1 ... -> b ...', b=num_samples)
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels , h // 8, w // 8]
        samples_cfg, intermediates = model.sample(
            ddim_steps,
            shape,
            cond,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        







