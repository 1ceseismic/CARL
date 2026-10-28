import torch as th

from carl import _carl
from carl.gymnasium.state import CARLBall, CARLCars, CARLResetState

from .initial import Box2D, Box3D, Euler, InAir, InitialState, OnSurface, Uniform
from ..events import ArenaSurface


def _check_side(side):
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")


def _surface_axes(surface, side):
    if surface == ArenaSurface.FLOOR:
        return (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    if surface == ArenaSurface.CEILING:
        return (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)
    if surface == ArenaSurface.SIDE_WALL:
        return (-float(side), 0.0, 0.0), (0.0, 1.0, 0.0)
    if surface == ArenaSurface.BACK_WALL:
        return (0.0, -float(side), 0.0), (1.0, 0.0, 0.0)
    raise ValueError("unsupported arena surface")


def _surface_position(batch, surface, position, side, rest):
    tangent = batch._vector(position, dimensions=2)
    if surface == ArenaSurface.FLOOR:
        depth = tangent.new_full((batch.count, 1), rest)
        return th.cat((tangent, depth), dim=-1)
    if surface == ArenaSurface.CEILING:
        depth = tangent.new_full((batch.count, 1), _carl.ARENA_MAX[2] - rest)
        return th.cat((tangent, depth), dim=-1)
    if surface == ArenaSurface.SIDE_WALL:
        x = tangent.new_full((batch.count, 1), side * (_carl.ARENA_MAX[0] - rest))
        return th.cat((x, tangent), dim=-1)
    if surface == ArenaSurface.BACK_WALL:
        y = tangent.new_full((batch.count, 1), side * (_carl.ARENA_MAX[1] - rest))
        return th.cat((tangent[:, :1], y, tangent[:, 1:]), dim=-1)
    raise ValueError("unsupported arena surface")


def _surface_forward(batch, normal, tangent, heading):
    angle = batch._scalar(heading)
    normal = th.tensor(normal, device=batch.device, dtype=th.float32).expand(batch.count, 3)
    tangent = th.tensor(tangent, device=batch.device, dtype=th.float32).expand(batch.count, 3)
    bitangent = th.linalg.cross(normal, tangent, dim=-1)
    return tangent * angle.cos()[:, None] + bitangent * angle.sin()[:, None]


class ResetSampler:

    def __init__(self, n_sim: int, seed: int = 0, device="cuda:0"):
        self.seed = seed & 0xFFFFFFFF
        self.device = th.device(device)
        self.generator = th.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
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
        return ResetBatch(indices, counts, self.seed, self.generator)


class ResetBatch:

    def __init__(self, simulation_indices: th.Tensor, reset_counts=None, seed: int = 0,
                 generator=None):
        self.simulation_indices = simulation_indices
        self.device = simulation_indices.device
        self.count = simulation_indices.numel()
        self.seed = seed & 0xFFFFFFFF
        self.generator = generator
        if self.generator is None:
            self.generator = th.Generator(device=self.device)
            self.generator.manual_seed(self.seed)
        self.reset_counts = (
            th.zeros_like(simulation_indices) if reset_counts is None else reset_counts
        )
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
        values = th.rand(
            (self.count, *shape),
            generator=self.generator,
            device=self.device,
            dtype=th.float32,
        )
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
            self.ego.on_surface(
                placement.surface, placement.position, placement.heading, placement.side
            )
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

    def on_surface(self, surface=ArenaSurface.FLOOR, position=(0, 0),
                   heading=0.0, side=1, **physics):
        _check_side(side)
        normal, tangent = _surface_axes(surface, side)
        position = _surface_position(self._batch, surface, position, side, _carl.CAR_REST_Z)
        forward = _surface_forward(self._batch, normal, tangent, heading)

        self._car.position.copy_(position)
        self._car.forward.copy_(forward)
        self._car.up.copy_(th.tensor(normal, device=self._batch.device, dtype=th.float32))
        self._car[..., _carl.OBS_ON_GROUND] = int(surface == ArenaSurface.FLOOR)
        return self.set(**physics)

    def on_floor(self, position=(0, 0), heading=0.0, **physics):
        return self.on_surface(ArenaSurface.FLOOR, position, heading, **physics)

    def on_ceiling(self, position=(0, 0), heading=0.0, **physics):
        return self.on_surface(ArenaSurface.CEILING, position, heading, **physics)

    def on_side_wall(self, position=(0, 0), heading=0.0, side=1, **physics):
        return self.on_surface(ArenaSurface.SIDE_WALL, position, heading, side, **physics)

    def on_back_wall(self, position=(0, 0), heading=0.0, side=1, **physics):
        return self.on_surface(ArenaSurface.BACK_WALL, position, heading, side, **physics)

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

    def on_surface(self, surface=ArenaSurface.FLOOR, position=(0, 0), side=1, **physics):
        _check_side(side)
        position = _surface_position(self._batch, surface, position, side, _carl.BALL_REST_Z)
        return self.set(position=position, **physics)

    def on_floor(self, position=(0, 0), **physics):
        return self.on_surface(ArenaSurface.FLOOR, position, **physics)

    def on_ceiling(self, position=(0, 0), **physics):
        return self.on_surface(ArenaSurface.CEILING, position, **physics)

    def on_side_wall(self, position=(0, 0), side=1, **physics):
        return self.on_surface(ArenaSurface.SIDE_WALL, position, side, **physics)

    def on_back_wall(self, position=(0, 0), side=1, **physics):
        return self.on_surface(ArenaSurface.BACK_WALL, position, side, **physics)
