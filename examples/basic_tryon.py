"""Basic tattoo virtual try-on example."""

from PIL import Image

from ttvton import PointPrompt, TatVTONPipeline

# Load images
body = Image.open("body.jpg")
tattoo = Image.open("tattoo.png")

# Define where to place the tattoo (click point on the body)
region = PointPrompt(coords=[(300, 400)])

# Run pipeline with defaults (12 GB GPU optimised)
pipe = TatVTONPipeline()
result = pipe(body_image=body, tattoo_image=tattoo, region=region)

# Save result
result.image.save("output.png")
print(f"Saved output.png (seed={result.seed})")
