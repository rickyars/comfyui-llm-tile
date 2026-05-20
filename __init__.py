"""
LLM-Tile - Generate tiled compositions based on LLM-generated prompts
"""

# Skip initialization if being imported as a test module by pytest
import sys
if __name__ != '__main__' and 'pytest' in sys.modules:
    # Don't execute relative imports when pytest is trying to import this as a test module
    pass
else:
    from .node import NODE_CLASS_MAPPINGS as TILE_NCM, NODE_DISPLAY_NAME_MAPPINGS as TILE_NDCM
    # Import your new advanced node mappings
    from .node_advanced import NODE_CLASS_MAPPINGS as ADV_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADV_NDCM
    # Import detailer node mappings
    from .node_detailer import NODE_CLASS_MAPPINGS as DET_NCM, NODE_DISPLAY_NAME_MAPPINGS as DET_NDCM
    from .node_detailer_adaptive import NODE_CLASS_MAPPINGS as ADET_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADET_NDCM

    # Combine all mappings
    NODE_CLASS_MAPPINGS = {**TILE_NCM, **ADV_NCM, **DET_NCM, **ADET_NCM}
    NODE_DISPLAY_NAME_MAPPINGS = {**TILE_NDCM, **ADV_NDCM, **DET_NDCM, **ADET_NDCM}

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

    # Version info
    __version__ = "0.3.0"  # Updated for regional diffusion support