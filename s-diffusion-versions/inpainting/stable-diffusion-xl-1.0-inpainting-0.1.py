"""
Stable-diffusion-xl-1.0-inpainting-0.1 model for inpainting area on image by it's mask.
Final number of steps is percentage calculation between num_inference_steps and strength.
When the strength parameter is set to 1 (i.e. starting in-painting from a fully masked image), the quality of the image is degraded.
The model retains the non-masked contents of the image, but images look less sharp.
"""

import torch
from PIL import Image
from datetime import datetime
from diffusers import AutoPipelineForInpainting

diff_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "black hand thumbs up"
image_path = r"D:\inpainting_test\hand_inpainting\216_camera_2.jpg"
mask_path = r"D:\inpainting_test\hand_inpainting\216_camera_2_mask.jpg"
# image_path = ""
# mask_path = ""
# prompt = ""
seed = 23

@torch.no_grad()
def main():
    pipe = AutoPipelineForInpainting.from_pretrained(diff_model,
                                                     torch_dtype=torch.float16,
                                                     variant="fp16").to(torch_device)

    image = Image.open(image_path).resize((1024, 1024))
    mask = Image.open(mask_path).resize((1024, 1024))

    generator = torch.Generator(device=torch_device).manual_seed(seed)

    image = pipe(
      prompt=prompt,
      image=image,
      mask_image=mask,
      guidance_scale=8.0,
      num_inference_steps=20,  # steps between 15 and 30 work well for us
      strength=0.79,  # make sure to use `strength` below 1.0
      generator=generator,
    ).images[0]

    image.save(f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png")


if __name__ == '__main__':
    main()
