# Import your node classes
from .node import TiledImageGenerator, TilePromptTemplateNode

# This is the critical part - create the NODE_CLASS_MAPPINGS at the top level of your package
NODE_CLASS_MAPPINGS = {
    "TiledImageGenerator": TiledImageGenerator,
    "TilePromptTemplateNode": TilePromptTemplateNode
}

# Add descriptions for the web UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGenerator": "Tiled Image Generator",
    "TilePromptTemplateNode": "Tile Prompt Template"
}

# Make these mappings available for importing
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]