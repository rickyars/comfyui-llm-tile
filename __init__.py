"""
LLM-Tile - Generate tiled compositions based on LLM-generated prompts
"""

import sys

try:
    from .node import NODE_CLASS_MAPPINGS as TILE_NCM, NODE_DISPLAY_NAME_MAPPINGS as TILE_NDCM
    from .node_advanced import NODE_CLASS_MAPPINGS as ADV_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADV_NDCM
    from .node_detailer_adaptive import NODE_CLASS_MAPPINGS as ADET_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADET_NDCM

    NODE_CLASS_MAPPINGS = {**TILE_NCM, **ADV_NCM, **ADET_NCM}
    NODE_DISPLAY_NAME_MAPPINGS = {**TILE_NDCM, **ADV_NDCM, **ADET_NDCM}
except ImportError:
    # Under pytest the modules are imported directly (no ComfyUI package
    # context), so the relative imports fail — that's fine for tests. In a
    # real ComfyUI process an import failure is a genuine error and must
    # surface, even if pytest happens to be loaded in the same process
    # (the old `'pytest' in sys.modules` guard silently unregistered the
    # whole pack in that case).
    if 'pytest' not in sys.modules:
        raise
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.4.0"
