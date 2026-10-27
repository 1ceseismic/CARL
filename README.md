# CARL

CUDA-Accelerated Rocket League (CARL) is a Rocket League physics simulator based on [RocketSim](https://github.com/ZealanL/RocketSim). Inspired by [PureJaxRL](https://chrislu.page/blog/meta-disco/), it is a vectorized environment for reinforcement learning in Rocket League that resides entirely on the GPU.

## Requirements

CARL requires Python 3.10 or newer, Linux, an NVIDIA GPU and driver, and the CUDA Toolkit for source builds.
Gymnasium, NumPy, and Torch are required dependencies.

Install the current source with:

```bash
uv sync
```

## Torch Environment

`CARLTorchVectorEnv` exposes each car as one actor. Cars remain grouped inside each physics simulation.

```python
import carl
import torch
from carl.gymnasium import CARLTorchVectorEnv

env = CARLTorchVectorEnv(
    n_sim=1024,
    n_blue=1,
    n_orange=1,
    frameskip=8,
    max_ticks=4096,
    no_touch_timeout_seconds=30.0,
    normalize=True,
)

observation = env.reset()
actions = torch.zeros((env.n_envs, len(carl.ACTION_NVECS)), dtype=torch.float32, device="cuda:0")
observation, reward, terminated, truncated, info = env.step(actions)
```

## Actions

Actions have shape `[n_envs, 7]` in Torch (`[n_sim, n_cars, 7]` natively) and are held for every tick within each `frameskip` window.

| # | Field | Values |
|---|-------|--------|
| 0 | horizontal | `[-1, 1]` |
| 1 | vertical | `[-1, 1]` |
| 2 | throttle | `[-1, 1]` |
| 3 | powerslide | `0`/`1` (`>= 0.5` active) |
| 4 | boost | `0`/`1` (`>= 0.5` active) |
| 5 | air roll | `[-1, 1]` |
| 6 | jump | `0`/`1` (`>= 0.5` active) |

## Observations

Observations have shape `[n_sim, n_cars, obs_dim]` (all `f32`). Car blocks are ordered observer first, then teammates, then opponents.

| Block | Fields |
|-------|--------|
| Ball | position, velocity, angular velocity |
| Cars | position, velocity, angular velocity, forward, up, boost, state flags |
| Boost pads | state, distance to observer |
| Self-to-ball | relative position, velocity |
| Self-to-car | relative position, velocity for each other car (same teammate/opponent order) |
| Goals | ball-to-own-goal and ball-to-opponent-goal, relative to the observing team |
| Ego | flip/jump availability (0/1), remaining dodge-window seconds |

Observation normalization is built-in and enabled via `normalize=True`.

Layout and physics constants such as `carl.OBS_BALL`, `carl.OBS_PER_CAR`, `carl.BOOST_PAD_POSITIONS`, `carl.PHYS_DT`, and `carl.REGULATION_TICKS` are available and help with downstream reward specification and other calculations.

## Custom Rewards

Rewards are customizable via the exposed `RewardContext`.

Reward functions receive a `RewardContext` and return `[n_sim, n_cars]`.

```python
def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / carl.CAR_MAX_SPEED

env = CARLTorchVectorEnv(1024, 1, 1, reward_funcs=[speed_reward])
```

The context contains previous state, current transition state, actions, score events, termination flags, and truncation flags. Reward functions are summed and multiplied by `reward_scale`. Custom rewards replace the default score reward.
