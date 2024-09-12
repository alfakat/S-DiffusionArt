"""
This script create canny image from given input image and prompt instructions
"""

import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime

from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

input_image = load_image("red_apples.png")
low_threshold = 100
high_threshold = 200
prompt = "oranges"

def create_canny_image(image):

    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    return canny_image

def main():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                                 torch_dtype=torch.float16,
                                                 use_safetensors=True)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    output = pipe(prompt=prompt,
                  image=create_canny_image(image=input_image)).images[0]

    output_image = Image.fromarray(np.array(output))
    output_image.save(f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png")


if __name__ == '__main__':
    main()
