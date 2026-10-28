from dataclasses import dataclass, field

from .initial import InitialState
from ..events import Condition


@dataclass
class Stage:

    event:        Condition
    within_ticks: int | None       = None
    fail_on:      list[Condition] = field(default_factory=list)

    def __post_init__(self):
        if self.within_ticks is not None and self.within_ticks <= 0:
            raise ValueError("within_ticks must be positive")


@dataclass
class Exercise:

    name:        str
    initial:     InitialState
    stages:      list[Stage]
    fail_on:     list[Condition] = field(default_factory=list)
    target_goal: str             = "positive_y"

    def __post_init__(self):
        if not self.stages:
            raise ValueError("an exercise needs at least one stage")


@dataclass
class TrainingPack:

    name:      str
    exercises: list[Exercise]

    def __post_init__(self):
        if not self.exercises:
            raise ValueError("a training pack needs at least one exercise")
