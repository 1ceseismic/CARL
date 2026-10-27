# Releases

**Status:** build tests and source-distribution creation pass. Wheel builds,
GPU testing, and publishing are not yet validated end-to-end.

## Wheel target

- Linux x86_64, glibc 2.35+, Python 3.11, CUDA 12.8.
- GPU architectures: 75, 80, 86, 89, 90; PTX: 90.
- Static CUDA runtime; no toolkit required for wheel users.
- NVIDIA driver 570.26+ and any newer driver required by Torch.

## Release setup

1. Add a trusted `carl-gpu` runner with Python 3.11, uv, and a compatible NVIDIA
   driver, but no system CUDA toolkit. Do not run untrusted code on it.
2. Configure the protected `pypi` environment and PyPI trusted publishing for
   `release.yml`. Confirm ownership of the package name.
3. Run `.github/workflows/release.yml` manually and verify the artifacts.
4. Tag `v<project.version>` to publish to PyPI and GitHub Releases after GPU tests.

The workflow builds from the sdist, repairs the wheel with auditwheel, checks
contents, and tests the installed simulator outside the checkout. Manual runs
do not publish. Local `uv build` does not repair or publish wheels.

## Tests

```bash
# Build checks; no CUDA required.
uv run --no-project --with setuptools python -m unittest discover -s tests -p 'test_build.py'

# Installed wheel; run outside the checkout on a real GPU.
python /absolute/path/to/CARL/tests/test_gpu.py
```
