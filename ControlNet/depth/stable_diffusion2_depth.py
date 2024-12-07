"""
This script predict depth map on input image and generate new image
with prompts instructions
"""

import torch
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionDepth2ImgPipeline

image_path = "red_apples.png"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

prompt = "tree oranges"
negative_propmt = "bad, deformed, ugly"

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
   "stabilityai/stable-diffusion-2-depth",
   torch_dtype=torch.float16).to(torch_device)

image = pipe(prompt=prompt,
             image=Image.open(image_path),
             negative_prompt=negative_propmt,
             strength=0.7).images[0]

image.save(f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png")

