import unittest

import torch as th

from carl import _carl


@unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
class NativeEventTests(unittest.TestCase):

    def setUp(self):
        self.env = _carl.Env(
            n_sim=2,
            n_blue=1,
            n_orange=0,
            seed=7,
            frameskip=8,
            capture_events=True,
            capture_ball_touches=True,
        )
        self.actions = th.zeros((2, 1, 7), device="cuda")

    def tensor(self, getter):
        owner = getter.__self__
        native = owner if isinstance(owner, _carl.EventState) else owner.events
        native.acquire(th.cuda.current_stream().cuda_stream)
        return th.from_dlpack(getter())

    def step(self):
        self.env.step_continuous(self.actions)
        return self.tensor(self.env.events.get_car_flags)

    def set_car(self, position, velocity=(0, 0, 0), rotation=(0, 0, 0, 1)):
        positions = th.tensor([[position]] * 2, device="cuda", dtype=th.float32)
        rotations = th.tensor([[rotation]] * 2, device="cuda", dtype=th.float32)
        velocities = th.tensor([[velocity]] * 2, device="cuda", dtype=th.float32)

        self.env.set_car(
            position=positions,
            rotation=rotations,
            velocity=velocities,
            angular_velocity=th.zeros((2, 1, 3), device="cuda"),
            demoed=th.zeros((2, 1), device="cuda", dtype=th.bool),
        )

    def set_ball(self, position, velocity=(0, 0, 0)):
        self.env.set_ball(
            position=th.tensor([position] * 2, device="cuda", dtype=th.float32),
            velocity=th.tensor([velocity] * 2, device="cuda", dtype=th.float32),
            angular_velocity=th.zeros((2, 3), device="cuda"),
        )

    def test_jump_is_accumulated_and_cleared_next_window(self):
        self.actions[..., -1] = 1
        flags = self.step()

        self.assertEqual(flags.shape, (2, 1))
        self.assertEqual(flags.dtype, th.int32)
        self.assertTrue(flags.is_cuda)
        self.assertEqual(((flags & _carl.EVENT_JUMP) != 0).tolist(), [[True], [True]])
        self.assertFalse((self.step() & _carl.EVENT_JUMP).any().item())

    def test_airborne_double_jump_and_dodge(self):
        for directional, event in ((0, _carl.EVENT_DOUBLE_JUMP), (1, _carl.EVENT_DODGE)):
            with self.subTest(directional=directional):
                self.env.reset()
                self.set_car((0, 0, 1000))
                self.actions[..., 1] = directional
                self.actions[..., -1] = 1
                flags = self.step()

                self.assertTrue(((flags & event) != 0).all().item())
                self.assertFalse((flags & _carl.EVENT_JUMP).any().item())
                self.assertEqual(((flags & event) != 0).sum().item(), 2)

    def test_reset_and_surface_baselines(self):
        flags = self.step()
        departures = _carl.EVENT_AIRBORNE | _carl.EVENT_LEAVE_WALL
        self.assertFalse((flags & departures).any().item())

        self.set_car((0, 0, 1000))
        flags = self.step()
        self.assertFalse((flags & departures).any().item())

        self.env.reset()
        self.assertFalse(self.tensor(self.env.events.get_car_flags).any().item())
        self.assertFalse(self.tensor(self.env.events.get_ball_flags).any().item())

    def test_ball_floor_contact_changes(self):
        self.env.frameskip = 64
        self.set_ball((0, 0, 300), (0, 0, -1000))
        self.step()

        entered = self.tensor(self.env.events.get_ball_surface_entered)
        left = self.tensor(self.env.events.get_ball_surface_left)
        flags = self.tensor(self.env.events.get_ball_flags)

        self.assertTrue(((entered & _carl.ARENA_FLOOR) != 0).all().item())
        self.assertTrue(((left & _carl.ARENA_FLOOR) != 0).all().item())
        self.assertTrue(th.equal(entered != 0, (flags & _carl.EVENT_BALL_SURFACE_ENTER) != 0))
        self.assertTrue(th.equal(left != 0, (flags & _carl.EVENT_BALL_SURFACE_LEAVE) != 0))

    def test_ball_wall_categories(self):
        self.env.frameskip = 32
        self.env.set_ball(
            position=th.tensor([[3980, 0, 1000], [0, 5000, 1000]], device="cuda", dtype=th.float32),
            velocity=th.tensor([[1000, 0, 0], [0, 1000, 0]], device="cuda", dtype=th.float32),
            angular_velocity=th.zeros((2, 3), device="cuda"),
        )
        self.step()
        entered = self.tensor(self.env.events.get_ball_surface_entered)

        self.assertTrue((entered[0] & _carl.ARENA_SIDE_WALL).item())
        self.assertTrue((entered[1] & _carl.ARENA_BACK_WALL).item())

    def test_ball_touch_flags_share_existing_contact_records(self):
        self.env.frameskip = 24
        self.set_car((0, 0, 17))
        self.set_ball((-300, 0, 90), (1500, 0, 0))
        flags = self.step()
        counts = self.tensor(self.env.events.get_ball_touch_counts)
        metadata = self.tensor(self.env.events.get_ball_touch_metadata)

        self.assertEqual(counts.tolist(), [1, 1])
        self.assertEqual(((flags & _carl.EVENT_BALL_TOUCH) != 0).sum().item(), 2)

        for sim in range(2):
            touch_tick = metadata[sim, 0, _carl.TOUCH_TICK].item()
            self.assertGreater(touch_tick, 0)
            self.assertLessEqual(touch_tick, 24)
            self.assertTrue((flags[sim, 0] & _carl.EVENT_BALL_TOUCH).item())

    def test_flip_reset_restores_spent_jump(self):
        self.env.frameskip = 1
        self.set_car((0, 0, 1000))
        self.actions[..., -1] = 1
        self.step()

        self.actions.zero_()
        self.set_car((0, 0, 1000), rotation=(1, 0, 0, 0))
        self.set_ball((0, 0, 1080))
        flags = self.step()

        self.assertTrue(((flags & _carl.EVENT_FLIP_RESET) != 0).all().item())
        self.assertFalse((self.step() & _carl.EVENT_FLIP_RESET).any().item())

    def test_demo_is_captured_on_victim(self):
        env = _carl.Env(1, 1, 1, 7, frameskip=4, capture_events=True)
        env.set_car(
            position=th.tensor([[[-50, 0, 17], [50, 0, 17]]], device="cuda", dtype=th.float32),
            rotation=th.tensor([[[0, 0, 0, 1], [0, 0, 1, 0]]], device="cuda", dtype=th.float32),
            velocity=th.tensor([[[2300, 0, 0], [-2300, 0, 0]]], device="cuda", dtype=th.float32),
            angular_velocity=th.zeros((1, 2, 3), device="cuda"),
            demoed=th.zeros((1, 2), device="cuda", dtype=th.bool),
        )
        env.step_continuous(th.zeros((1, 2, 7), device="cuda"))
        flags = self.tensor(env.events.get_car_flags)

        self.assertEqual(((flags & _carl.EVENT_DEMOED) != 0).sum().item(), 1)

    def test_goal_survives_autoreset(self):
        self.set_ball((0, 5300, 200))
        self.step()
        flags = self.tensor(self.env.events.get_ball_flags)
        sides = self.tensor(self.env.events.get_goal_side)

        self.assertTrue(((flags & _carl.EVENT_GOAL) != 0).all().item())
        self.assertEqual(sides.tolist(), [1, 1])
        self.assertEqual(((flags & _carl.EVENT_GOAL) != 0).sum().item(), 2)
        self.assertTrue(self.tensor(self.env.get_dones).all().item())

    def test_disabled_capture_and_independent_touch_capture(self):
        env = _carl.Env(1, 1, 0, 7, capture_ball_touches=True)

        with self.assertRaises(ValueError):
            self.tensor(env.events.get_car_flags)

        counts = self.tensor(env.events.get_ball_touch_counts)
        self.assertEqual(counts.tolist(), [0])
        self.assertEqual(counts.data_ptr(), self.tensor(env.get_ball_touch_counts).data_ptr())

    def test_capture_does_not_change_physics(self):
        plain = _carl.Env(2, 1, 0, 7, frameskip=8)

        for _ in range(3):
            plain.step_continuous(self.actions)
            self.step()

        th.cuda.synchronize()
        expected = self.tensor(plain.get_state)
        actual = self.tensor(self.env.get_state)
        self.assertTrue(th.equal(expected, actual))

    def test_flag_storage_is_independent_of_frameskip(self):
        flags = self.tensor(self.env.events.get_car_flags)
        pointer = flags.data_ptr()
        self.env.frameskip = 64
        resized = self.tensor(self.env.events.get_car_flags)

        self.assertEqual(resized.shape, (2, 1))
        self.assertEqual(resized.data_ptr(), pointer)

    def test_speed_change_is_peak_tick_gain(self):
        single = _carl.Env(2, 1, 0, 7, frameskip=1, capture_events=True)
        position = th.tensor([[0, 0, 1000]] * 2, device="cuda", dtype=th.float32)
        velocity = th.tensor([[0, 0, -100]] * 2, device="cuda", dtype=th.float32)
        angular = th.zeros((2, 3), device="cuda")

        self.env.set_ball(position, velocity, angular)
        single.set_ball(position, velocity, angular)
        self.step()
        gains = []

        for _ in range(8):
            single.step_continuous(self.actions)
            gains.append(self.tensor(single.events.get_ball_speed_change).clone())

        peak = th.stack(gains).amax(dim=0)
        aggregate = self.tensor(self.env.events.get_ball_speed_change)

        th.testing.assert_close(aggregate, peak)
        self.assertTrue((aggregate > 0).all().item())

    def test_selected_setters_leave_other_simulation_untouched(self):
        selected = th.tensor([0], device="cuda", dtype=th.int64)
        self.set_car((0, 0, 1000))
        self.env.set_car(
            position=th.tensor([[[0, 0, 17]]], device="cuda", dtype=th.float32),
            rotation=th.tensor([[[0, 0, 0, 1]]], device="cuda", dtype=th.float32),
            velocity=th.zeros((1, 1, 3), device="cuda"),
            angular_velocity=th.zeros((1, 1, 3), device="cuda"),
            demoed=th.zeros((1, 1), device="cuda", dtype=th.bool),
            simulation_indices=selected,
        )
        self.env.set_ball(
            position=th.tensor([[0, 5300, 200]], device="cuda", dtype=th.float32),
            velocity=th.zeros((1, 3), device="cuda"),
            angular_velocity=th.zeros((1, 3), device="cuda"),
            simulation_indices=selected,
        )
        self.env.set_match_state(
            blue_score=th.tensor([2], device="cuda", dtype=th.int32),
            orange_score=th.tensor([1], device="cuda", dtype=th.int32),
            episode_ticks=th.tensor([10], device="cuda", dtype=th.int32),
            simulation_indices=selected,
        )
        self.env.reset_boost_pads(simulation_indices=selected)
        self.step()

        flags = self.tensor(self.env.events.get_ball_flags)
        score = self.tensor(self.env.get_transition_score_difference)

        self.assertEqual(((flags & _carl.EVENT_GOAL) != 0).tolist(), [True, False])
        self.assertEqual(score.tolist(), [2, 0])
