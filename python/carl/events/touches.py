import torch as th

from dataclasses import dataclass, field
from .. import _carl


@dataclass
class BallTouchBatch:

    counts:   th.Tensor
    metadata: th.Tensor
    values:   th.Tensor
    _native:  _carl.EventState | None = field(default=None, repr=False, compare=False)

    def wait(self):
        if self._native is not None:
            self._native.acquire(th.cuda.current_stream().cuda_stream)
        return self

    def snapshot(self):
        self.wait()
        return BallTouchBatch(self.counts.clone(), self.metadata.clone(), self.values.clone())

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
        return self.relative_speed_after - self.relative_speed_before

    @property
    def absolute_speed_change(self) -> th.Tensor:
        return self.speed_change.abs()

    @property
    def solver_ball_velocity_change(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_SOLVER_DELTA_X:_carl.TOUCH_EXTRA_DELTA_X]

    @property
    def extra_ball_velocity_change(self) -> th.Tensor:
        return self.values[..., _carl.TOUCH_EXTRA_DELTA_X:_carl.TOUCH_VALUE_SIZE]

    @property
    def ball_velocity_change(self) -> th.Tensor:
        return self.solver_ball_velocity_change + self.extra_ball_velocity_change
