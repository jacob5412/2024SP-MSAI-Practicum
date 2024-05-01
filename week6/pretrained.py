from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", use_safetensors=True
)

pipeline.save_pretrained("model")
