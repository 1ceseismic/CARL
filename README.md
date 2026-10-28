# CARL

CUDA-Accelerated Rocket League (CARL) is a Rocket League physics simulator based on [RocketSim](https://github.com/ZealanL/RocketSim). Inspired by [PureJaxRL](https://chrislu.page/blog/meta-disco/), it is a vectorized environment for reinforcement learning in Rocket League that resides entirely on the GPU.

## Requirements

- Linux, Python 3.10+, and an NVIDIA GPU with a CUDA 12.8-compatible driver.
- Source builds require the CUDA 12.8 toolkit and GCC 12.

## Building from Source

Prebuilt wheels are not yet published. With CUDA 12.8 and GCC 12 installed, run from the repository root:

```bash
uv sync
```

The build uses `CUDA_HOME` when set, as the toolkit installer normally does, and otherwise searches for `nvcc` on `PATH`.

Set `CARL_CUDA_ARCHITECTURES` (for example, `89`) if `nvidia-smi` is unavailable.

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

Rewards are customizable via the exposed `RewardContext`.

Reward functions receive a `RewardContext` and return `[n_sim, n_cars]`.

```python
import carl

def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / carl.CAR_MAX_SPEED

env = CARLTorchVectorEnv(1024, 1, 1, reward_funcs=[speed_reward])
```

The context contains previous state, current transition state, actions, score events, termination flags, and truncation flags. Reward functions are summed and multiplied by `reward_scale`. Custom rewards replace the default score reward.
