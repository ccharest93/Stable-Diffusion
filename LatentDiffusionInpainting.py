from torch import nn
from Unet import Unet
from AutoencoderKL import AutoencoderKL
import open_clip
import torch
import numpy as np
from einops import repeat    
from functools import partial
from tqdm import tqdm
from PIL import Image
import torch
import time
#from pytorch_memlab import MemReporter
import sys
import gradio as gr

#Helper functions
def to_device(obj, device):
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    else:
        return obj

class GPUManager():
    def __init__(self, tensors, device, task):
        self.tensors = tensors
        self.device = device
        self.task = task
        self.total_time = 0
        #self.reporter = MemReporter()
    
    def __enter__(self):
        start = time.perf_counter()
        self.tensors = to_device(self.tensors, self.device)
        print("ENTERING...")
        print("Context: {}".format(self.task))
        print("Device: {}".format(self.device))
        #self.reporter.report()
        self.total_time += time.perf_counter() - start
        return self.tensors
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        start = time.perf_counter()
        self.total_time += time.perf_counter() - start
        print("EXITING...")
        print("Context: {}".format(self.task))
        print("Time taken: {}".format(self.total_time))
        #self.reporter.report()

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
        "image": repeat(image, "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask, "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image, "1 ... -> n ...", n=num_samples),
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


#Latent Diffusion Inpainting Model
class LatentDiffusionInpainting(nn.Module):
    def __init__(self):
        super().__init__()
        #diffusion model initialization
        self.diffusion_model = Unet()
        self.diffusion_model.eval()
        self.dpm_num_timesteps = 1000
        self.diffusion_param_schedule()
        self.concat_keys = ('mask', 'masked_image')
        self.masked_image_key = 'masked_image'

        #autoencoder model initialization
        self.scale_factor = 0.18215
        self.channels = 4
        self.first_stage_model = AutoencoderKL()
        self.first_stage_model.eval()

        #clip model initialization
        self.cond_stage_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", device=torch.device('cpu'), pretrained="laion2b_s32b_b79k")
        del self.cond_stage_model.visual
        self.cond_stage_model.eval()

    def diffusion_param_schedule(self):
        linear_start=1e-4
        linear_end=2e-2

        alphas = 1. - (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, 1000, dtype=torch.float64) ** 2).numpy()
        alphas_cumprod = np.cumprod(alphas, axis=0)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
    
    @torch.no_grad()
    def sample(self,
               num_ddim_timesteps,
               shape,
               cond,
               eta= 1.0,
               temperature=1.0,
               x_T=None,
               log_every_t=100, 
               unconditional_guidance_scale=1.0, 
               unconditional_conditioning=None):
        
        #get diffusion timesteps
        c = self.dpm_num_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, self.dpm_num_timesteps , c)))
        ddim_timesteps += 1

        # calculations for variance schedule
        alphacums = self.alphas_cumprod.cpu()
        self.ddim_alphas = alphacums[ddim_timesteps] #BUG if ddim_timestep 1000 is needed
        self.ddim_alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
        self.ddim_sigmas = eta * np.sqrt((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev))
        self.ddim_sqrt_one_minus_alphas  = np.sqrt(1. - self.ddim_alphas)

        #Diffusion loop
        b = shape[0]
        x = x_T
        timesteps = ddim_timesteps
        intermediates = {'x_inter': [x], 'pred_x0': [x]}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        with GPUManager(tensors={"diffusion_model":self.diffusion_model,
                                "x":x,
                                "cond":cond,
                                "unconditional_conditioning":unconditional_conditioning,
                                "sqrt_alphas_cumprod":self.sqrt_alphas_cumprod,
                                "sqrt_one_minus_alphas_cumprod":self.sqrt_one_minus_alphas_cumprod},device= torch.device('cuda'),task= "Diffusion Loop") as tensors:
            for i, step in enumerate(tqdm(time_range, desc='DDIM Sampler', total=total_steps)):
                #prepare inputs
                index = total_steps - i - 1
                batch_size, *_, device = *tensors['x'].shape, tensors['x'].device
                tensors['t_in'] = torch.full((b,), step, device=torch.device('cuda'), dtype=torch.long)
                tensors['x_in'] = torch.cat([tensors['x']] * 2)
                tensors['t_in'] = torch.cat([tensors['t_in']] * 2)
                tensors['c_in'] = dict()
                for k in tensors['cond']:
                    if isinstance(tensors['cond'][k], list):
                        tensors['c_in'][k] = [torch.cat([
                            tensors['unconditional_conditioning'][k][i],
                            tensors['cond'][k][i]]) for i in range(len(tensors['cond'][k]))]
                    else:
                        tensors['c_in'][k] = torch.cat([
                                tensors['unconditional_conditioning'][k],
                                tensors['cond'][k]])
                tensors['xc'] = torch.cat([tensors['x_in']] + tensors['c_in']['c_concat'], dim=1)
                del tensors['x_in']
                tensors['cc'] = torch.cat(tensors['c_in']['c_crossattn'], 1)
                del tensors['c_in']

                #diffusion model forward pass
                tensors['model_uncond'], tensors['model_t']  = tensors['diffusion_model'](tensors['xc'], timesteps=tensors['t_in'], context=tensors['cc']).chunk(2)
                tensors['model_output'] = tensors['model_uncond'] + unconditional_guidance_scale * (tensors['model_t'] - tensors['model_uncond'])
                del tensors['model_uncond'], tensors['model_t']
                
                # select parameters corresponding to the currently considered timestep
                tensors['a_t'] = torch.full((batch_size, 1, 1, 1), self.ddim_alphas[index], device=device)
                tensors['a_prev'] = torch.full((batch_size, 1, 1, 1), self.ddim_alphas_prev[index], device=device)
                tensors['sigma_t'] = torch.full((batch_size, 1, 1, 1), self.ddim_sigmas[index], device=device)
                tensors['sqrt_one_minus_at'] = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=device)

                #remove predicted noise from the image for the current timestep
                tensors['pred_x0'] = (tensors['x'] - tensors['sqrt_one_minus_at'] * tensors['model_output']) / tensors['a_t'].sqrt()
                tensors['dir_xt'] = (1. - tensors['a_prev'] - tensors['sigma_t']**2).sqrt() * tensors['model_output']
                del tensors['model_output']
                noise = tensors['sigma_t'] * noise_like(tensors['x'].shape, device, False) * temperature
                tensors['x'] = tensors['a_prev'].sqrt() * tensors['pred_x0'] + tensors['dir_xt'] + noise

                #logging of intermediate results
                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(tensors['x'].to(torch.device('cpu')))
                    intermediates['pred_x0'].append(tensors['pred_x0'].to(torch.device('cpu')))
                del tensors['pred_x0'], tensors['dir_xt'], tensors['a_t'], tensors['a_prev'], tensors['sigma_t'], tensors['sqrt_one_minus_at']
            self.diffusion_model = tensors['diffusion_model'].to(torch.device('cpu'))
            out = tensors['x']
        
        #autoencoder decoding
        out = 1. / self.scale_factor * out
        with GPUManager(tensors={"self.first_stage_model":self.first_stage_model,
                                 "out":out},device= torch.device('cuda'),task= "KLAutoEncoder decoding") as tensors:
            out = tensors["self.first_stage_model"].decode(tensors["out"]).to(torch.device('cpu'))
            self.first_stage_model = tensors["self.first_stage_model"].to(torch.device('cpu'))
        result = torch.clamp((out + 1.0) / 2.0,min=0.0, max=1.0)
        result = result.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255
        return [Image.fromarray(img.astype(np.uint8)) for img in result]
    
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

    def load_checkpoints(self, ckpt_path = "512-inpainting-ema.ckpt"):
        di = torch.load(ckpt_path)["state_dict"]
        di = {k.replace("model.diffusion_model", "diffusion_model",1): v for k, v in di.items()}
        di = {k.replace(".in_layers.0.", ".norm1.",1): v for k, v in di.items()}
        di = {k.replace(".in_layers.2.", ".conv1.",1): v for k, v in di.items()}
        di = {k.replace(".out_layers.0.", ".norm2.",1): v for k, v in di.items()}
        di = {k.replace(".out_layers.3.", ".conv2.",1): v for k, v in di.items()}
        di = {k.replace(".0.op.", ".conv.",1): v for k, v in di.items()}
        di = {k.replace(".input_blocks.0.0.", ".input_blocks.0.",1): v for k, v in di.items()}
        di = {k.replace(".nin_shortcut.", ".skip_connection.",1): v for k, v in di.items()}
        di = {k.replace("cond_stage_model.model.", "cond_stage_model.",1): v for k, v in di.items()}
        di = {k.replace(".emb_layers.1.", ".emb_proj.",1): v for k, v in di.items()}
        self.load_state_dict(di, strict=False)

#functions required by GRadio
def init_model(ckpt_path):
    model = LatentDiffusionInpainting()
    model.load_checkpoints(ckpt_path)
    return model

@torch.no_grad()
def inpaint(input, prompt, ddim_steps=45, num_samples=1, scale=10, seed=100):
    #preprocess
    init_image = input["image"].convert("RGB")
    init_mask = input["mask"].convert("RGB")
    image = pad_image(init_image) # resize to integer multiple of 32
    mask = pad_image(init_mask) # resize to integer multiple of 32
    w, h = image.size
    batch = make_batch_sd(image, mask, txt=prompt, num_samples=num_samples)

    #encoding text conditioning
    c = model.get_conditional(batch["txt"])

    #encoding image conditioning
    c_cat = list()
    for ck in model.concat_keys:
        cc = batch[ck].float()
        if ck != model.masked_image_key:
            bchw = [num_samples, 4, h // 8, w // 8]
            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
        else:
            with GPUManager(tensors= { "model.first_stage_model":model.first_stage_model,
                                        "cc":cc}, device= torch.device('cuda'),task= "KLAutoEncoder encoding") as tensors:
                cc = tensors["model.first_stage_model"].encode(tensors["cc"]).sample().to(torch.device('cpu'))
                model.first_stage_model = tensors["model.first_stage_model"].to(torch.device('cpu'))
            cc = model.scale_factor * cc
        c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)
    cond = {"c_concat": [c_cat], "c_crossattn": [c]}

    #encoding null prompt conditioning
    uc_cross = model.get_conditional("")
    uc_cross = repeat(uc_cross, '1 ... -> b ...', b=num_samples)
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

    #sampling
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=torch.device('cpu'),dtype=torch.float32)
    shape = [num_samples, model.channels , h // 8, w // 8]
    result = model.sample(
        ddim_steps,
        shape,
        cond,
        eta=1.0,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc_full,
        x_T=start_code,
    )
    return result

with torch.no_grad():
    model = init_model(sys.argv[1])
    #model = init_model(r'checkpoints\512-inpainting-ema.ckpt')
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## Stable Diffusion Inpainting")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', tool='sketch', type="pil")
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    num_samples = gr.Slider(
                        label="Images", minimum=1, maximum=4, value=4, step=1)
                    ddim_steps = gr.Slider(label="Steps", minimum=1,
                                        maximum=50, value=45, step=1)
                    scale = gr.Slider(
                        label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
            with gr.Column():
                gallery = gr.Gallery(label="Generated images", show_label=False).style(
                    grid=[2], height="auto")

        run_button.click(fn=inpaint, inputs=[input_image, prompt, ddim_steps, num_samples, scale, seed], outputs=[gallery])


    block.launch()









