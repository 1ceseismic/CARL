import os
from pathlib import Path

_arena_obj = Path(__file__).with_name("assets") / "arena.obj"
if not _arena_obj.is_file():
    _arena_obj = Path(__file__).resolve().parents[2] / "assets" / "arena.obj"
os.environ.setdefault("CARL_ARENA_OBJ", str(_arena_obj))

try:
    from . import _carl
except ImportError as exc:
    raise ImportError(
        "CARL requires its native CUDA extension. Install a compatible wheel "
        "or build with the CUDA 12.8 toolkit using `uv sync`."
    ) from exc

__all__ = [name for name in vars(_carl) if name == "Env" or name.isupper()]
globals().update({name: getattr(_carl, name) for name in __all__})
