from diffusers import StableDiffusionPipeline

model_id = "sd-legacy/stable-diffusion-v1-5"

print("Downloading model...")

pipe = StableDiffusionPipeline.from_pretrained(model_id)

print("Model downloaded successfully!")