from .node import TiledImageGenerator
from .node_flux import TiledFluxGenerator

NODE_CLASS_MAPPINGS = {
    "TiledImageGenerator": TiledImageGenerator,
    "TiledFluxGenerator": TiledFluxGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGenerator": "Tiled Image Generator (Improved)",
    "TiledFluxGenerator": "Tiled Flux Generator (Inpainting)"
}