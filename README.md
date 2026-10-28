# CARL

CUDA-Accelerated Rocket League (CARL) is a Rocket League physics simulator based on [RocketSim](https://github.com/ZealanL/RocketSim). Inspired by [PureJaxRL](https://chrislu.page/blog/meta-disco/), it is a vectorized environment for reinforcement learning in Rocket League that resides entirely on the GPU.

## Requirements

CARL requires Linux, Python 3.10+, an NVIDIA GPU, and a driver compatible with CUDA 12.8 and your Torch build. Gymnasium, NumPy, and Torch are installed automatically. The native simulation is required; there is no Python-only mode.

## Building from Source

Install the **CUDA 12.8 development toolkit** and a compatible C++ compiler (for example GCC 12), then:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
uv sync
```

This builds the local simulation. Toolkit discovery also checks `nvcc` on `PATH` and standard installation locations. Set `CARL_CUDA_ARCHITECTURES` to override detected GPU targets.

Prebuilt wheels are planned, not yet published.

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
