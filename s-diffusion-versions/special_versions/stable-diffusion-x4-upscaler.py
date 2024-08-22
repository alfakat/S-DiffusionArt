"""
This scripts using stable-diffusion-x4-upscaler for increase the quality of image.
There is no parameters as steps, seeds, image resolution in pipe.
Script runs with stable-diffusion-v1-4 model, text-to-image pipe.
"""

import torch
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionUpscalePipeline

prompt = ""
image_path = ""

diff_model = "stabilityai/stable-diffusion-x4-upscaler"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        diff_model, variant="fp16", torch_dtype=torch.float16)

    pipe.to(torch_device)
    pipe.enable_attention_slicing()

    input_image = Image.open(image_path)
    input_image.resize((128, 128))

    image = pipe(prompt=prompt, image=input_image,  negative_prompt=prompt).images[0]
    image.save(f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png")


if __name__ == '__main__':
    main()