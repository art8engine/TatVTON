# TatVTON — Tattoo Virtual Try-On

Realistic tattoo compositing on body photos using **SAM 2** mask extraction and **SDXL Inpainting + ControlNet + IP-Adapter** synthesis.

## Installation

```bash
pip install -e .
```

With DensePose support (optional):

```bash
pip install -e ".[densepose]"
```

Development tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from PIL import Image
from ttvton import TatVTONPipeline, PointPrompt

body = Image.open("body.jpg")
tattoo = Image.open("tattoo.png")

pipe = TatVTONPipeline()
result = pipe(
    body_image=body,
    tattoo_image=tattoo,
    region=PointPrompt(coords=[(300, 400)]),
)
result.image.save("output.png")
```

## Configuration

Three levels of configuration:

| Level | Method | Example |
|-------|--------|---------|
| Defaults | `TatVTONPipeline()` | 12 GB GPU optimised |
| Config | `TatVTONPipeline(TatVTONConfig(resolution=768))` | Session-wide |
| Per-call | `pipe(..., strength=0.95, seed=42)` | Call-specific |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NVIDIA GPU with 12+ GB VRAM (recommended)

## License

MIT
