"""
This scripts shows the capabilities of stable-diffusion-xl-base-1.0
for higher resolution and quality.
There are steps and negative prompts as parameters.
"""

import torch
from datetime import datetime
from diffusers import DiffusionPipeline

prompt = "an apple tree in desert"
negative_prompt = 'anime, cartoon'

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                             torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to(torch_device)

    images = pipe(prompt=prompt,
                  num_inference_steps=20,
                  negative_prompt=negative_prompt).images[0]
    images.save(f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png")


if __name__ == '__main__':
    main()