# Prevents pytest from walking up to the parent ComfyUI pytest.ini,
# which would cause relative import errors in __init__.py
import sys
import os

# Add the project root to sys.path so tests can import from utils/ etc.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def pytest_collection(session):
    """Remove root __init__.py from collection."""
    # Prevents the root __init__.py from being imported as a test module
    pass
