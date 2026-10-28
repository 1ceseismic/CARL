import math

import torch as th

from carl import _carl
from carl.gymnasium.state import CARLBall, CARLCars, CARLResetState

from .initial import Box2D, Box3D, Euler, InAir, InitialState, OnSurface, Uniform
from ..events import ArenaSurface


class ResetSampler:

    def __init__(self, n_sim: int, seed: int = 0, device="cuda:0"):
        self.seed = seed & 0xFFFFFFFF
        self.device = th.device(device)
        self.reset_counts = th.zeros(n_sim, dtype=th.int64, device=self.device)

    def select(self, selected: th.Tensor) -> "ResetBatch":
        if selected.device != self.reset_counts.device or selected.ndim != 1:
            raise ValueError("reset selection must be one-dimensional on the sampler device")

        if selected.dtype == th.bool:
            if selected.shape != self.reset_counts.shape:
                raise ValueError("reset mask must have one entry per simulation")
            indices = selected.nonzero(as_tuple=True)[0]
        elif selected.dtype == th.int64:
            indices = selected
        else:
            raise TypeError("reset selection must be a bool mask or int64 indices")

        counts = self.reset_counts[indices].clone()
        self.reset_counts[indices] += 1
        return ResetBatch(indices, counts, self.seed)


class ResetBatch:

    def __init__(self, simulation_indices: th.Tensor, reset_counts=None, seed: int = 0):
        self.simulation_indices = simulation_indices
        self.device = simulation_indices.device
        self.count = simulation_indices.numel()
        self.seed = seed & 0xFFFFFFFF
        self.reset_counts = (
            th.zeros_like(simulation_indices) if reset_counts is None else reset_counts
        )
        self._draw = 0
        self._ball = CARLBall.from_tensor(
            th.zeros((self.count, _carl.OBS_BALL), device=self.device, dtype=th.float32)
        )
        self._cars = CARLCars.from_tensor(
            th.zeros((self.count, 1, _carl.OBS_PER_CAR), device=self.device, dtype=th.float32), 1
        )
        self.ego = CarReset(self)
        self.ball = BallReset(self)
        self.ego.on_floor(position=(0, -1000), boost=100)
        self.ball.on_floor()

    def uniform(self, low=0.0, high=1.0, shape=()) -> th.Tensor:
        width = math.prod(shape)
        draws = th.arange(self._draw, self._draw + width, device=self.device, dtype=th.int64)
        self._draw += width

        key = ((self.simulation_indices[:, None] + 1) * 0x9E3779B9) & 0xFFFFFFFF
        key = key ^ ((self.reset_counts[:, None] * 0x85EBCA6B) & 0xFFFFFFFF)
        key = key ^ (((draws + 1) * 0xC2B2AE35) & 0xFFFFFFFF) ^ self.seed
        key = ((key ^ (key >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
        key = ((key ^ (key >> 15)) * 0x846CA68B) & 0xFFFFFFFF
        key = key ^ (key >> 16)

        values = (key >> 8).to(th.float32) * (1.0 / (1 << 24))
        values = values.reshape(self.count, *shape)
        return low + (high - low) * values

    def choice(self, values) -> th.Tensor:
        values = th.as_tensor(values, device=self.device)
        if values.shape[0] == 0:
            raise ValueError("choice needs at least one value")
        indices = (self.uniform() * values.shape[0]).long()
        return values[indices]

    def _scalar(self, value) -> th.Tensor:
        if isinstance(value, Uniform):
            return self.uniform(value.low, value.high)
        value = th.as_tensor(value, device=self.device, dtype=th.float32)
        return th.broadcast_to(value, (self.count,))

    def _vector(self, value, dimensions=3) -> th.Tensor:
        if isinstance(value, (Box2D, Box3D)):
            low = th.tensor(value.low, device=self.device, dtype=th.float32)
            high = th.tensor(value.high, device=self.device, dtype=th.float32)
            return low + (high - low) * self.uniform(shape=(dimensions,))
        if isinstance(value, th.Tensor):
            value = value.to(device=self.device, dtype=th.float32)
            return th.broadcast_to(value, (self.count, dimensions))
        if len(value) != dimensions:
            raise ValueError(f"expected {dimensions} vector components")
        return th.stack([self._scalar(component) for component in value], dim=-1)

    def sample(self, initial: InitialState) -> CARLResetState:
        placement = initial.ego.placement
        if isinstance(placement, OnSurface):
            if placement.surface != ArenaSurface.FLOOR:
                raise NotImplementedError("only floor placement is implemented")
            self.ego.on_floor(placement.position, placement.heading)
        elif isinstance(placement, InAir):
            self.ego.in_air(placement.position, placement.orientation)
        else:
            raise TypeError("unsupported car placement")

        self.ego.set(
            velocity=initial.ego.velocity,
            angular_velocity=initial.ego.angular_velocity,
            boost=initial.ego.boost,
        )
        self.ball.set(
            position=initial.ball.position,
            velocity=initial.ball.velocity,
            angular_velocity=initial.ball.angular_velocity,
        )
        return self.build()

    def build(self) -> CARLResetState:
        internal = th.zeros(
            (self.count, 1, _carl.CAR_INTERNAL_SIZE), device=self.device, dtype=th.float32
        )
        internal[..., _carl.CAR_INT_ON_GROUND] = self._cars[..., _carl.OBS_ON_GROUND]

        return CARLResetState(
            simulation_indices = self.simulation_indices,
            ball               = self._ball,
            cars               = self._cars,
            car_internal_state = internal,
        )


class CarReset:

    def __init__(self, batch: ResetBatch):
        self._batch = batch
        self._car = batch._cars.ego

    def set(self, *, position=None, orientation=None, velocity=None,
            angular_velocity=None, boost=None):
        batch = self._batch
        if position is not None:
            self._car.position.copy_(batch._vector(position))
            self._car[..., _carl.OBS_ON_GROUND] = 0
        if orientation is not None:
            self._car[..., _carl.OBS_ON_GROUND] = 0
            roll = batch._scalar(orientation.roll)
            pitch = batch._scalar(orientation.pitch)
            yaw = batch._scalar(orientation.yaw)
            cr, sr = roll.cos(), roll.sin()
            cp, sp = pitch.cos(), pitch.sin()
            cy, sy = yaw.cos(), yaw.sin()

            self._car.forward.copy_(th.stack((cy * cp, sy * cp, -sp), dim=-1))
            up = th.stack((
                cy * sp * cr + sy * sr,
                sy * sp * cr - cy * sr,
                cp * cr,
            ), dim=-1)
            self._car.up.copy_(up)
        if velocity is not None:
            self._car.velocity.copy_(batch._vector(velocity))
        if angular_velocity is not None:
            self._car.angular_velocity.copy_(batch._vector(angular_velocity))
        if boost is not None:
            self._car.boost.copy_(batch._scalar(boost))
        return self

    def on_floor(self, position=(0, 0), heading=0.0, **physics):
        xy = self._batch._vector(position, dimensions=2)
        height = xy.new_full((self._batch.count, 1), _carl.CAR_REST_Z)
        self.set(position=th.cat((xy, height), dim=-1), orientation=Euler(yaw=heading), **physics)
        self._car[..., _carl.OBS_ON_GROUND] = 1
        return self

    def in_air(self, position, orientation=None, **physics):
        orientation = Euler() if orientation is None else orientation
        return self.set(position=position, orientation=orientation, **physics)

    def local_vector(self, forward=0.0, left=0.0, up=0.0) -> th.Tensor:
        batch = self._batch
        side = th.linalg.cross(self._car.up, self._car.forward, dim=-1)
        return (
            self._car.forward * batch._scalar(forward)[:, None]
            + side * batch._scalar(left)[:, None]
            + self._car.up * batch._scalar(up)[:, None]
        )

    def local_point(self, forward=0.0, left=0.0, up=0.0) -> th.Tensor:
        return self._car.position + self.local_vector(forward, left, up)


class BallReset:

    def __init__(self, batch: ResetBatch):
        self._batch = batch
        self._ball = batch._ball

    def set(self, *, position=None, velocity=None, angular_velocity=None):
        values = (
            ("position", position),
            ("velocity", velocity),
            ("angular_velocity", angular_velocity),
        )
        for name, value in values:
            if value is not None:
                getattr(self._ball, name).copy_(self._batch._vector(value))
        return self

    def on_floor(self, position=(0, 0), **physics):
        xy = self._batch._vector(position, dimensions=2)
        height = xy.new_full((self._batch.count, 1), _carl.BALL_REST_Z)
        return self.set(position=th.cat((xy, height), dim=-1), **physics)
