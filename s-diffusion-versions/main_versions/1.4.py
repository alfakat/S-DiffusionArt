import torch

from typing import List
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms as tfms

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import LMSDiscreteScheduler

# move to config
# read from config
prompt = ['An astronaut riding a horse'] # ['An astronaut riding a horse']
input_image = [r'C:\Users\Katya\PycharmProjects\blink_synthetic\test\astronaut_rides_horse.png'] # r'C:\Users\Katya\PycharmProjects\blink_synthetic\test\astronaut_rides_horse.png'
batch_size = 1
generator = torch.manual_seed(50)
height = 512 # recommended
width = 512 # recommended
num_steps = 50
guidance_scale = 7.5


diff_model = "runwayml/stable-diffusion-v1-4"
#diff_model = "stabilityai/stable-diffusion-xl-base-1.0"
clip_model = "openai/clip-vit-large-patch14"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# autoencoder part
vae = AutoencoderKL.from_pretrained(diff_model, subfolder="vae", force_upcast=False).to(torch_device)

# tokenizer and text encoder part
tokenizer = CLIPTokenizer.from_pretrained(clip_model)
text_encoder = CLIPTextModel.from_pretrained(clip_model).to(torch_device)
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt")
uncond_input = tokenizer([""]*batch_size, padding = "max_length", max_length = text_input.input_ids.shape[-1], # 77
                        return_tensors="pt")
text_encoder = text_encoder.to(torch_device)
# print(tokenizer.get_vocab())  # Shows the vocabulary
print(tokenizer.encoder)

# UNet architecture part
unet = UNet2DConditionModel.from_pretrained(diff_model, subfolder="unet").to(torch_device)

# Scheduler part
scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
scheduler.set_timesteps(num_steps)


@torch.no_grad()
def pil_to_latent(input_image: List) -> torch:
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)

    for im in input_image:
        latent = vae.encode(tfms.ToTensor()(Image.open(im)).unsqueeze(0).to(torch_device)*2-1)
        return 0.18215 * latent.latent_dist.sample()


def step_size(num_steps) -> torch:
    """Input:
    Output
    Function to calculate step 'jumps' while generating"""
    start_value, end_value = 999.0000, 0.0000
    step_size = (start_value - end_value) / (num_steps - 1)

    tensor_values = torch.tensor([start_value - i * step_size for i in range(num_steps)])

    return tensor_values


@torch.no_grad()
def text_embeddings(text_input, uncond_input) -> torch:
    """Input
    Output
    Coding prompt embeddings"""

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


@torch.no_grad()
def main():
    if input_image:
        latents = pil_to_latent(input_image)
        latents = latents * scheduler.init_noise_sigma
        for t in tqdm(step_size(num_steps)):
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            prediction = unet(latent_model_input,
                              t,
                              encoder_hidden_states=text_embeddings(text_input, uncond_input))['sample']

            noise_pred_uncond, noise_pred_text = prediction.chunk(2)
            prediction = noise_pred_uncond * (noise_pred_text - noise_pred_uncond)  # for noise add 7.5 noise

            latents = scheduler.step(prediction, t, latents).prev_sample
    else:
        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator).to(torch_device)

        latents = latents * scheduler.init_noise_sigma
        for t in tqdm(step_size(num_steps)):
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            prediction = unet(latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings(text_input, uncond_input))['sample']

            noise_pred_uncond, noise_pred_text = prediction.chunk(2)
            prediction = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(prediction, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save('output_Katya3.png')


if __name__ == '__main__':
    main()





