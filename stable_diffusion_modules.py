"""
This script showed the pure Stable Diffusion pipe included 3 main modules: Variational autoencoder VAE,
Prompt (text) tokenization, Convolutional neural network U-NET.
It doesn't have additional uncoditional guidance noize, only controlled by prompt and steps.
Script running on stable-diffusion-v1-4 model.
"""

import os
import torch

from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import LMSDiscreteScheduler


prompt = 'an apple tree in a desert'
batch_size = 1
generator = torch.manual_seed(4)
height = 512 # recommended
width = 512 # recommended
num_steps = 50


diff_model = "CompVis/stable-diffusion-v1-4"
clip_model = "openai/clip-vit-large-patch14"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# autoencoder part
vae = AutoencoderKL.from_pretrained(diff_model, subfolder="vae", force_upcast=False).to(torch_device)

# tokenizer and text encoder part
tokenizer = CLIPTokenizer.from_pretrained(clip_model)
text_encoder = CLIPTextModel.from_pretrained(clip_model).to(torch_device)
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt")

# UNet architecture part
unet = UNet2DConditionModel.from_pretrained(diff_model, subfolder="unet").to(torch_device)

# Scheduler part
scheduler = LMSDiscreteScheduler.from_pretrained(diff_model, subfolder="scheduler")
scheduler.set_timesteps(num_steps)


def step_size(num_steps: int) -> torch:
    """Denoising process
    Input: wished number of denoising scopes
    Output: tensor
    """
    start_value, end_value = 999.0000, 0.0000
    step_size = (start_value - end_value) / (num_steps - 1)

    tensor_values = torch.tensor([start_value - i * step_size for i in range(num_steps)])

    return tensor_values


@torch.no_grad()
def text_embeddings(text_input) -> torch:
    """Converting tokens to embeddings
    Input: prompt text in tokens
    Output: vector in unet format
    """

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    return text_embeddings


@torch.no_grad()
def main():
    latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator).to(torch_device)

    latents = latents * scheduler.init_noise_sigma
    for t in tqdm(step_size(num_steps)):

        latent_model_input = scheduler.scale_model_input(latents, t)

        prediction = unet(latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings(text_input))['sample']

        latents = scheduler.step(prediction, t, latents).prev_sample
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    Image.fromarray((image[0] * 255).round().astype("uint8")).save(
        os.path.join(".\outputs", f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png"))


if __name__ == '__main__':
    main()
