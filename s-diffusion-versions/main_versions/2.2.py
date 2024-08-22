"""
This scripts shows the capabilities of improved second Stable Diffusion model.
There is still no parameters as steps, seeds, image resolution in pipe.
Script runs with stable-diffusion-2 model, text-to-image pipe.
"""

import os
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

diff_model = "stabilityai/stable-diffusion-2"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    scheduler = EulerDiscreteScheduler.from_pretrained(diff_model, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(diff_model, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)

    prompt = "an apple tree in a desert"
    image = pipe(prompt=prompt).images[0]

    image.save(os.path.join("..\..\outputs", f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png"))


if __name__ == '__main__':
    main()
