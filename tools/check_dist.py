"""Check that releases contain the simulator, Python API and arena asset."""

import sys
import tarfile
import zipfile
from pathlib import Path


def check(directory):
    wheels  = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    assert wheels and sources, "Expected both wheels and a source distribution"
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert any(
                name.startswith("carl/_carl.") and name.endswith(".so")
                for name in names
            ), f"Native simulator missing from {wheel}"
            for required in (
                "carl/__init__.py",
                "carl/gymnasium/torch.py",
                "carl/assets/arena.obj",
            ):
                assert required in names, f"{required} missing from {wheel}"
            assert not any(":Zone.Identifier" in name for name in names)
    for source in sources:
        with tarfile.open(source) as archive:
            names = {name.partition("/")[2] for name in archive.getnames()}
            for required in (
                "setup.py",
                "CMakeLists.txt",
                "python/carl.cu",
                "src/Kernels.cu",
                "include/RLEnvironment.cuh",
                "assets/arena.obj",
            ):
                assert required in names, f"{required} missing from {source}"
    print("Distribution contents verified")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
