import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))
prompt = "a white cat"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("upsampled_cat.png")

# prompt (str or List[str], optional) — prompt to be encoded device — (torch.device): torch device
# num_images_per_prompt (int) — number of images that should be generated per prompt
# do_classifier_free_guidance (bool) — whether to use classifier free guidance or not
# negative_prompt (str or List[str], optional) — The prompt or prompts not to guide the image generation. If not defined, one has to pass negative_prompt_embeds instead. Ignored when not using guidance (i.e., ignored if guidance_scale is less than 1).
# prompt_embeds (torch.Tensor, optional) — Pre-generated text embeddings. Can be used to easily tweak text inputs, e.g. prompt weighting. If not provided, text embeddings will be generated from prompt input argument.
# negative_prompt_embeds (torch.Tensor, optional) — Pre-generated negative text embeddings. Can be used to easily tweak text inputs, e.g. prompt weighting. If not provided, negative_prompt_embeds will be generated from negative_prompt input argument.
# lora_scale (float, optional) — A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
# clip_skip (int, optional) — Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
# Encodes the prompt into text encoder hidden states.