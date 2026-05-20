# Prevents pytest from walking up to the parent ComfyUI pytest.ini,
# which would cause relative import errors in __init__.py
import sys
import os

# Stub out comfy.* so pure-function tests can import node_detailer_adaptive
# without needing a live ComfyUI installation
from types import ModuleType
import unittest.mock

def _make_comfy_stubs():
    comfy = ModuleType('comfy')
    comfy.sample = ModuleType('comfy.sample')
    comfy.model_management = ModuleType('comfy.model_management')
    comfy.samplers = ModuleType('comfy.samplers')
    comfy.utils = ModuleType('comfy.utils')

    # KSampler stub with empty lists (enough for INPUT_TYPES to not crash)
    ksample = ModuleType('comfy.samplers.KSampler')
    ksample.SAMPLERS = []
    ksample.SCHEDULERS = []
    comfy.samplers.KSampler = ksample

    # ProgressBar stub
    comfy.utils.ProgressBar = unittest.mock.MagicMock

    sys.modules['comfy'] = comfy
    sys.modules['comfy.sample'] = comfy.sample
    sys.modules['comfy.model_management'] = comfy.model_management
    sys.modules['comfy.samplers'] = comfy.samplers
    sys.modules['comfy.utils'] = comfy.utils

_make_comfy_stubs()

# Add the project root to sys.path so tests can import from utils/ etc.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def pytest_collection(session):
    """Remove root __init__.py from collection."""
    # Prevents the root __init__.py from being imported as a test module
    pass
