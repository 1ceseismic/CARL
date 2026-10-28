# CARL

CUDA-Accelerated Rocket League (CARL) is a Rocket League physics simulator based on [RocketSim](https://github.com/ZealanL/RocketSim). Inspired by [PureJaxRL](https://chrislu.page/blog/meta-disco/), it is a vectorized environment for reinforcement learning in Rocket League that resides entirely on the GPU.

## Requirements

- Linux and Python 3.10 or newer.
- An NVIDIA GPU with a driver that supports the CUDA runtime used by the installed PyTorch build. CUDA 12.8 requires Linux driver 570.26 or newer.
- For source builds: the **CUDA 12.8 development toolkit** and GCC 12, or another compiler supported by that toolkit.

`uv sync` installs Gymnasium, NumPy, and PyTorch automatically.

## Building from Source

Prebuilt wheels are not yet published, so install from a checkout. The build compiles the CUDA extension locally and statically links the CUDA runtime; the GPU driver is only needed at run time.

Install the CUDA 12.8 development toolkit and GCC 12 (or another compiler supported by that toolkit), then run the following from the repository root:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
uv sync
```

`CUDA_HOME` must contain `bin/nvcc`, the CUDA headers, and `libcudart_static.a`. If it is unset, the build looks for `nvcc` on `PATH` and in standard locations such as `/usr/local/cuda-12.8`.

CUDA architectures are detected with `nvidia-smi`. To build without a GPU, or to target specific devices, set `CARL_CUDA_ARCHITECTURES` (for example, `89`).

## Torch Environment

`CARLTorchVectorEnv` exposes each car as one actor. Cars remain grouped inside each physics simulation.

```python
import torch
from carl.gymnasium import CARLTorchVectorEnv

env = CARLTorchVectorEnv(n_sim=8192, n_blue=1, n_orange=1)

observation = env.reset()

actions = torch.zeros(
    env.action_space.shape,
    dtype=torch.float32,
    device=env.device
)

observation, reward, terminated, truncated, info = env.step(actions)
```

## Custom Rewards

Rewards are customizable via the exposed `StepContext`.

Reward functions receive a `StepContext` and return `[n_sim, n_cars]`.

```python
import carl

def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / carl.CAR_MAX_SPEED

env = CARLTorchVectorEnv(1024, 1, 1, reward_funcs=[speed_reward])
```

The context contains previous state, current transition state, actions, score events, termination flags, and truncation flags. Reward functions are summed and multiplied by `reward_scale`. Custom rewards replace the default score reward.

## Events

```python
import torch
from carl.gymnasium import CARLTorchVectorEnv

env = CARLTorchVectorEnv(
    n_sim=1024, n_blue=1, n_orange=0, frameskip=8,
    capture_events=True, capture_ball_touches=True,
)
env.reset()
actions = torch.zeros(env.action_space.shape, device=env.device)
actions[:, -1] = 1
env.step(actions)

events = env.events                 # Also context.events in rewards
jumped = events.ego.jumped           # bool [simulation], over the frameskip window
touched = events.ego.ball_touched    # bool [simulation]
scored = events.ball.scored          # bool [simulation]
touches = events.ball_touches        # Detailed contacts; mask with touches.valid
saved = events.snapshot()            # Retain GPU data beyond the next step/reset

env.close()
```
