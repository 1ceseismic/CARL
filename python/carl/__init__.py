import os
from pathlib import Path

_asset = Path(__file__).with_name("assets") / "arena.obj"
if not _asset.is_file():
    _asset = Path(__file__).resolve().parents[2] / "assets" / "arena.obj"
os.environ.setdefault("CARL_ARENA_OBJ", str(_asset))

from . import _carl

# Re-export the native layout and physics constants without copying their values
__all__ = [name for name in vars(_carl) if name == "Env" or name.isupper()]
globals().update({name: getattr(_carl, name) for name in __all__})
