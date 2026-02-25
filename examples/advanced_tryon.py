"""Advanced tattoo try-on — custom config, DensePose, per-call overrides."""

from PIL import Image

from ttvton import BBoxPrompt, TatVTONConfig, TatVTONPipeline

# Custom configuration for 24 GB GPU
config = TatVTONConfig(
    resolution=1024,
    num_inference_steps=40,
    strength=0.9,
    controlnet_conditioning_scale=0.6,
    ip_adapter_scale=0.7,
    offload_strategy="none",  # keep everything on GPU
    use_densepose=True,
    seed=42,
)

pipe = TatVTONPipeline(config)

body = Image.open("body.jpg")
tattoo = Image.open("tattoo.png")

# Use bounding box to specify the tattoo area
region = BBoxPrompt(bbox=(200, 300, 500, 600))

# Per-call overrides
result = pipe(
    body_image=body,
    tattoo_image=tattoo,
    region=region,
    strength=0.95,
    ip_adapter_scale=0.8,
    seed=123,
    prompt="a photorealistic dragon tattoo on skin, vivid colors",
)

result.image.save("output_advanced.png")
print(f"Saved output_advanced.png (seed={result.seed})")

# Clean up GPU memory
pipe.unload()
