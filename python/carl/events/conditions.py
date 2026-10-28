from dataclasses import dataclass

from .enums import ArenaSurface, ContactSurface


class Condition:

    pass


@dataclass
class BallTouch(Condition):

    surface:        ContactSurface | None = None
    min_height:     float | None          = None
    min_speed_gain: float | None          = None


@dataclass
class Jump(Condition):

    pass


@dataclass
class DoubleJump(Condition):

    pass


@dataclass
class Dodge(Condition):

    pass


@dataclass
class FlipReset(Condition):

    pass


@dataclass
class LeaveSurface(Condition):

    pass


@dataclass
class Grounded(Condition):

    pass


@dataclass
class Airborne(Condition):

    pass


@dataclass
class EnterWall(Condition):

    pass


@dataclass
class LeaveWall(Condition):

    pass


@dataclass
class Demoed(Condition):

    pass


@dataclass
class BallSurfaceEnter(Condition):

    surface: ArenaSurface | None = None


@dataclass
class BallSurfaceLeave(Condition):

    surface: ArenaSurface | None = None


@dataclass
class BallContactGround(Condition):

    pass


@dataclass
class BallAccelerates(Condition):

    by: float


@dataclass
class Goal(Condition):

    pass
