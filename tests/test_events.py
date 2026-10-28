import unittest

import torch as th

from carl import _carl
from carl.events import (
    ArenaSurface, BallSurfaceEnter, BallTouch, BallTouchBatch, ContactSurface,
    CarEvents, Event, Jump,
)


class EventDefinitionTests(unittest.TestCase):

    def test_integer_identifiers(self):
        events = th.tensor(list(Event), dtype=th.int32)

        self.assertEqual(events.unique().numel(), len(Event))
        self.assertEqual(Event.BALL_TOUCH, _carl.EVENT_BALL_TOUCH)
        self.assertEqual(ContactSurface.UNDERSIDE, _carl.TOUCH_UNDERSIDE)

    def test_standalone_conditions(self):
        touch = BallTouch(surface=ContactSurface.FRONT, min_speed_gain=500)
        contact = BallSurfaceEnter(surface=ArenaSurface.FLOOR)

        self.assertEqual(touch.surface, ContactSurface.FRONT)
        self.assertEqual(contact.surface, ArenaSurface.FLOOR)
        self.assertEqual(Jump(), Jump())

    def test_named_flags_are_elementwise(self):
        events = CarEvents(th.tensor([
            [Event.JUMP | Event.BALL_TOUCH, Event.DODGE],
            [Event.FLIP_RESET, Event.JUMP],
        ], dtype=th.int32))

        self.assertEqual(events.jumped.tolist(), [[True, False], [False, True]])
        self.assertEqual(events.ball_touched.tolist(), [[True, False], [False, False]])
        self.assertEqual(events.flip_reset.tolist(), [[False, False], [True, False]])

    def test_touch_batch(self):
        batch = BallTouchBatch(
            counts   = th.tensor([1, 3], dtype=th.int32),
            metadata = th.zeros((2, 2, _carl.TOUCH_METADATA_SIZE), dtype=th.int32),
            values   = th.zeros((2, 2, _carl.TOUCH_VALUE_SIZE)),
        )
        batch.metadata[0, 0, _carl.TOUCH_TICK] = 12
        batch.metadata[0, 0, _carl.TOUCH_SURFACE] = ContactSurface.FRONT
        batch.values[..., _carl.TOUCH_SOLVER_DELTA_X] = 100
        batch.values[..., _carl.TOUCH_EXTRA_DELTA_X] = 50

        self.assertEqual(batch.valid.tolist(), [[True, False], [True, True]])
        self.assertEqual(batch.overflow.tolist(), [False, True])
        self.assertEqual(batch.tick[0, 0].item(), 12)
        self.assertEqual(batch.surface[0, 0].item(), ContactSurface.FRONT)
        self.assertEqual(batch.ball_velocity_change[0, 0].tolist(), [150, 0, 0])
