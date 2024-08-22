import os
import torch
from datetime import datetime
from diffusers import StableDiffusionInpaintPipeline

prompt = ""
image_path = ""
mask_image_path = ""

diff_model = "runwayml/stable-diffusion-inpainting"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        diff_model,
        revision="fp16",
        torch_dtype=torch.float16)

    image = pipe(prompt=prompt, image=image_path, mask_image=mask_image_path).images[0]
    image.save(os.path.join("..\..\outputs", f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png"))


if __name__ == '__main__':
    main()