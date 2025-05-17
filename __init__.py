"""
LLM-Tile - Generate tiled compositions based on LLM-generated prompts
"""
from .node import NODE_CLASS_MAPPINGS as TILE_NCM, NODE_DISPLAY_NAME_MAPPINGS as TILE_NDCM
# Import your new advanced node mappings
from .node_advanced import NODE_CLASS_MAPPINGS as ADV_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADV_NDCM

# Combine all mappings (excluding the flux node)
NODE_CLASS_MAPPINGS = {**TILE_NCM, **ADV_NCM}
NODE_DISPLAY_NAME_MAPPINGS = {**TILE_NDCM, **ADV_NDCM}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "0.2.0"  # Adjust version as needed