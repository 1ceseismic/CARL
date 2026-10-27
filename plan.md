# CARL Events and Training Packs

**Goal:** A `Skill` defines a sampled starting state and an ordered sequence of measurable stages. Complete and check each step before starting the next.

1. **Agree on the public contract.** Use physics ticks for deadlines; `repetitions` counts distinct events after a stage becomes active. Global `Skill.fail_on` and active `Stage.fail_on` both end the attempt. An event on the last allowed tick passes before timeout is checked.

   ```python
   skill = Skill(
       initial_state=sample_wall_setup,
       fail_on=[Event.DEMOED],
       stages=[
           Stage(Event.BALL_TOUCH, car=0, within_ticks=90),
           Stage(Event.LEAVE_WALL, car=0, within_ticks=45),
           Stage(Event.BALL_TOUCH, car=0, repetitions=3, within_ticks=240),
           Stage(Event.FLIP_RESET, car=0, within_ticks=120,
                 fail_on=[Event.DODGE]),
           Stage(Event.DODGE, car=0, within_ticks=60),
           Stage(Event.BALL_TOUCH, car=0, within_ticks=60),
           Stage(Event.GOAL, team="blue", within_ticks=240),
       ],
   )
   ```

   **Check:** Confirm event meanings and whether end-of-`frameskip` terminal observations are sufficient.

2. **Capture measured ball touches.** Add opt-in, ordered physics-tick recording and expose `BALL_TOUCH` through `carl.events`. Record the touching car, car-local contact point/normal, depth, pre/post ball–car relative speed, signed/absolute speed change, and ball velocity change. Derive an approximate front/rear/side/roof/underside label from the normal. Fix the existing `wheelSource` allocation.

   **Check:** Rotated-car contacts report the right local surface; continuous contact counts once, while separate touches within one `frameskip` count separately.

3. **Add core events.** Record jump, double jump, dodge, groundedness and wall changes, ball-wheel flip reset, demo, goal, and ball-surface contact changes. Expose the same batch in `RewardContext.events`.

   **Check:** Events arrive in physics order; starting in contact with a surface creates no spurious touch or departure event.

4. **Build initial-state setters and sampling.** `ResetBatch` supplies selected simulations and seeded GPU randomness. Its builders output the existing `CARLResetState`.

   ```python
   def sample_wall_setup(s: ResetBatch) -> CARLResetState:
       side = s.choice(("left", "right"))
       car = s.car(0).on_wall(
           side=side, y=s.uniform(-1500, -900),
           height=s.uniform(700, 1100),
           heading_deg=s.uniform(-15, 15),
           speed=s.uniform(900, 1200),
           boost=s.uniform(60, 100),
       )
       s.ball.on_wall(
           side=side, near=car.local_point(forward=260),
       ).launch(
           speed=s.uniform(350, 650),
           azimuth_deg=s.uniform(75, 105),
           elevation_deg=s.uniform(-8, 12),
       )
       return s.build()
   ```

   Provide `on_floor`, `on_wall`, `on_ceiling`, and `on_backboard` for both car and ball; generic car Euler and ball position/velocity setters; and `car.local_point/vector`. Backboard “own/opponent” takes a reference car. Ball angles describe **velocity direction**, since CARL does not store ball orientation.

   **Check:** Seeded samples are reproducible, correctly positioned on each surface, and physically plausible.

5. **Implement the Skill runner.** Track stage, repetition count, and elapsed ticks per simulation on the GPU. Support contact-surface and speed-change filters, global/stage failures, deadline failure, final-stage success, and failure on a premature goal.

   **Check:** Synthetic event streams cover repetitions, same-tick order, and exact-deadline completion.

6. **Integrate termination and reset.** Retain native autoreset for native endings. Add `reset_selected(mask)` for skill-only endings, preserve pre-reset observations from both paths, and call the Skill sampler for every ended simulation. Skill success/failure terminates; external environment limits truncate. Report outcome, reason, and final stage in `info`.

   **Check:** Each ended simulation resets once; unrelated simulations and their progress remain untouched.

7. **Encode Skill observations.** With `skill_observation=True`, return physics observations plus padded, ordered stage tokens and a mask. Tokens include objectives, filters, failure rules, passed/active status, repetition progress, and time remaining; retain terminal tokens in final observations.

   **Check:** Tokens update on stage passage and reset, while environments without a Skill retain their existing observation format.

8. **Verify and document the complete example.** Run the flip-reset shot in 1v0 and 1v1, test each failure path and surface setter, and measure opt-in event overhead.

   **Check:** The documented example works end to end before considering additional events or setters.
