"""
This scripts shows the capabilities of very first Stable Diffusion model.
There is no parameters as steps, seeds, image resolution in pipe.
Script runs with stable-diffusion-v1-4 model, text-to-image pipe.
"""

import os
import torch

from datetime import datetime
from diffusers import StableDiffusionPipeline

diff_model = "CompVis/stable-diffusion-v1-4"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    pipe = StableDiffusionPipeline.from_pretrained(diff_model,
                                                   torch_dtype=torch.float16,
                                                   use_safetensors=True)
    pipe = pipe.to(torch_device)

    prompt = "an apple tree in a desert"
    image = pipe(prompt=prompt).images[0]

    image.save(os.path.join("..\..\outputs", f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png"))


if __name__ == '__main__':
    main()