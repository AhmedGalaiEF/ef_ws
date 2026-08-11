from __future__ import annotations

import sys
import importlib.util
from pathlib import Path


OPENAIAPI_ROOT = Path(__file__).resolve().parents[1]
if str(OPENAIAPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENAIAPI_ROOT))


def pytest_ignore_collect(collection_path, config):
    """Keep optional agent tests out of collection when pydantic is absent."""
    if collection_path.name == "test_dependency.py":
        return False
    if collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return False
    if importlib.util.find_spec("pydantic") is None:
        return True
    try:
        import pydantic
    except Exception:
        return True
    return not pydantic.VERSION.startswith("2")
