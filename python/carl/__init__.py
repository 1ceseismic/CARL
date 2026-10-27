import os
from pathlib import Path

_asset = Path(__file__).with_name("assets") / "arena.obj"
if not _asset.is_file():
    _asset = Path(__file__).resolve().parents[2] / "assets" / "arena.obj"
os.environ.setdefault("CARL_ARENA_OBJ", str(_asset))

try:
    from . import _carl
except ImportError as exc:
    raise ImportError(
        "CARL requires its native CUDA simulation extension. Install a compatible "
        "CARL wheel, or build from source with the CUDA 12.8 development toolkit "
        "and `uv sync`. The original loader error is shown above."
    ) from exc

# Re-export the native layout and physics constants without copying their values
__all__ = [name for name in vars(_carl) if name == "Env" or name.isupper()]
globals().update({name: getattr(_carl, name) for name in __all__})
