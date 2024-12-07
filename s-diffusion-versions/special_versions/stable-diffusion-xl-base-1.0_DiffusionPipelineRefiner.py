from diffusers import DiffusionPipeline
import torch

# - from diffusers import StableDiffusionXLPipeline
# + from optimum.intel import OVStableDiffusionXLPipeline
#
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# - pipeline = StableDiffusionXLPipeline.from_pretrained(model_id)
# + pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)
# prompt = "A majestic lion jumping from a big stone at night"
# image = pipeline(prompt).images[0]

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.enable_model_cpu_offload()
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

image.save('output_Katya4.png')


# image = (images / 2 + 0.5).clamp(0, 1)
# image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
# images = (image * 255).round().astype("uint8")
# pil_images = [Image.fromarray(image) for image in images]
# pil_images[0].save('output_Katya3.png')
