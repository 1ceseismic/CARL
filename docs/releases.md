# Releases and testing

CARL is one package: the native CUDA simulation and Python API are both required.
Prebuilt wheels are planned for PyPI; they are not yet published.

## Initial wheel target

- Linux x86_64, glibc 2.35+, CPython 3.11, CUDA 12.8.
- Native GPU code: 75, 80, 86, 89, 90; PTX: 90 for forward compatibility.
- Static CUDA runtime: no system toolkit or `nvcc` needed to use a wheel.
- NVIDIA driver 570.26+ as a conservative CUDA 12.8 baseline. Also satisfy
  Torch's driver requirement: a CUDA 13 Torch build requires a newer driver.

Source builds use a system CUDA 12.8 toolkit and compatible host compiler,
such as GCC 12. Discovery checks `CUDA_HOME`, `nvcc` on `PATH`, then standard
locations. An invalid explicit `CUDA_HOME` fails rather than falling back.
`uv sync` inside this repository builds the checkout, not a published wheel.

## Release workflow

`.github/workflows/release.yml` builds a source distribution and builds the wheel
from that archive in a CUDA development container. It uses auditwheel to inspect
and repair the wheel, verifies required files, and tests the installed wheel on
a real GPU outside the checkout. Manual runs validate without publishing.
Matching `v<project.version>` tags publish to PyPI and mirror distributions to
GitHub Releases only after GPU testing succeeds.

Before enabling releases:

1. Configure a dedicated trusted self-hosted runner labeled `carl-gpu`, with
   Linux x86_64, Python 3.11, uv, and a driver compatible with CARL and Torch.
   Use a host without a system toolkit to verify wheel independence. Never run
   untrusted pull-request code on this runner.
2. Configure a protected GitHub environment named `pypi` and a PyPI trusted
   publisher for this repository and `release.yml`. Confirm ownership or
   availability of the distribution name before publishing.
3. Run the workflow manually and inspect the artifacts before tagging a release.

For equivalent local release architecture selection:

```bash
export CARL_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"
uv build
```

Local `uv build` does not repair wheels or publish them; use the release workflow
for distributable artifacts. Extend the wheel matrix only after validating each
additional Python/platform/CUDA combination.

## Tests

Build-configuration tests do not need CUDA:

```bash
uv run --no-project --with setuptools python -m unittest discover -s tests -p 'test_build.py'
```

After installing a built wheel, run the GPU smoke test from outside the checkout:

```bash
python /absolute/path/to/CARL/tests/test_gpu.py
```

It checks arena assets, reset, step, and Torch tensors on a real GPU. Missing
simulation or GPU support fails the test rather than skipping it.
