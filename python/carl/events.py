import torch as th

from dataclasses import dataclass
from enum import IntEnum

from . import _carl


class Event(IntEnum):
    BALL_TOUCH = _carl.EVENT_BALL_TOUCH


class ContactSurface(IntEnum):
    FRONT = _carl.TOUCH_FRONT
    REAR = _carl.TOUCH_REAR
    LEFT = _carl.TOUCH_LEFT
    RIGHT = _carl.TOUCH_RIGHT
    ROOF = _carl.TOUCH_ROOF
    UNDERSIDE = _carl.TOUCH_UNDERSIDE


@dataclass(frozen=True)
class BallTouchBatch:
    """GPU arrays [simulation, touch]; use ``valid`` to exclude empty slots."""

    counts: th.Tensor
    metadata: th.Tensor
    values: th.Tensor

    @property
    def valid(self) -> th.Tensor:
        slot = th.arange(self.metadata.shape[1], device=self.counts.device)
        return slot[None, :] < self.counts[:, None]

    @property
    def overflow(self) -> th.Tensor:
        return self.counts > self.metadata.shape[1]

    @property
    def tick(self) -> th.Tensor:
        return self.metadata[..., _carl.TOUCH_TICK]

    @property
    def car(self) -> th.Tensor:
        return self.metadata[..., _carl.TOUCH_CAR]

    @property
    def surface(self) -> th.Tensor:
        return self.metadata[..., _carl.TOUCH_SURFACE]

    @property
    def point_local(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_POINT_X:_carl.TOUCH_NORMAL_X]

    @property
    def normal_local(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_NORMAL_X:_carl.TOUCH_DEPTH]

    @property
    def depth(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_DEPTH]

    @property
    def relative_speed_before(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_REL_SPEED_BEFORE]

    @property
    def relative_speed_after(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_REL_SPEED_AFTER]

    @property
    def speed_change(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_SPEED_CHANGE]

    @property
    def absolute_speed_change(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_ABS_SPEED_CHANGE]

    @property
    def solver_ball_velocity_change(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_SOLVER_DELTA_X:_carl.TOUCH_EXTRA_DELTA_X]

    @property
    def extra_ball_velocity_change(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_EXTRA_DELTA_X:_carl.TOUCH_VALUE_SIZE]

    @property
    def ball_velocity_change(self) -> th.Tensor:
        """Solver change plus requested extra impulse, before any speed cap."""
        return self.solver_ball_velocity_change + self.extra_ball_velocity_change


def get_ball_touches(env: _carl.Env) -> BallTouchBatch:
    """Copy the most recent step's event batch from a capture-enabled environment."""
    counts = th.from_dlpack(env.get_ball_touch_counts())
    metadata = th.from_dlpack(env.get_ball_touch_metadata())
    values = th.from_dlpack(env.get_ball_touch_values())

    th.cuda.synchronize(counts.device)
    batch = BallTouchBatch(counts.clone(), metadata.clone(), values.clone())
    th.cuda.current_stream(counts.device).synchronize()
    return batch


__all__ = ["Event", "ContactSurface", "BallTouchBatch", "get_ball_touches"]
