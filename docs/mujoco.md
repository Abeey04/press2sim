# MuJoCo baseline: why our own simulator

The same block was built in MuJoCo 4.0 (`press2sim/mujoco_foam.py`) and scored exactly like our model:
same presses, same dots, same input. Run it after `model` with
`python -m press2sim mujoco --trials trials.csv --raw raw` (needs `pip install mujoco`).

![The real H1 press, our simulated block and MuJoCo's, all driven by the same measured cap depth](../media/H1_vs_mujoco.gif)

**Setup:** a soft-body ("flex") grid of 24 × 14 × 8 vertices over the whole block, with the fitted
Young's modulus and our dashpot's 1.24 ms as its damping, its base pinned to the table, and the cap as a kinematic box that follows
the measured cap depth. Poisson's ratio was picked the way ours was, on the training presses (0.3 beat 0).
Making it run at all took two fixes:

- **Integrator:** the real block is so light (78 g) for its stiffness that its elastic waves are fast,
  and MuJoCo's cheaper integrators blew up within milliseconds. It runs with the implicit `discrete`
  integrator at 1 ms steps, with the foam's mass scaled up 100× and gravity off. The presses are
  still much slower than the block's elastic response, so this doesn't change them.
- **Contact:** MuJoCo's soft contacts give way against a light, stiff body. With default settings the
  cap went straight through the foam. Short contact time constants, and driving the cap by where the
  foam surface under it actually is, fixed that: up to about 11 mm deep the surface under the cap follows
  the measured depth within 0.1 mm. Deeper than that it doesn't, for the reason in point 2.

**What happened:**

1. **Up to about 11 mm deep, the two solvers agree.** Given the same physics (linear elastic block,
   rigid table, no soft top), MuJoCo and our solver predict the dots within 7–17% of each other, which
   cross-checks both.
2. **Past about 15 mm, MuJoCo's block turns itself inside out under the cap.** Its default material
   (Saint Venant–Kirchhoff, which MuJoCo's documentation describes as meant for large rotations with
   small strains) stops resisting once it is squashed hard. The elements under the cap collapse until
   their top sits below the layer beneath, and the foam surface under the cap sinks 4–5 mm below the cap
   itself: 20 mm deep with the cap at 16 mm, 34 mm deep with it at 29 mm. `results/model/mujoco_comparison.json`
   lists the vertex heights under the cap at each depth. Every real press in this session went 20–50 mm deep.
   Our own model is not free of this: its 3 mm soft top layer is squeezed flat under the cap and ends up
   slightly inside out too, by under a millimetre (see [model.md](model.md), Limits). The difference is
   one of degree, and ours comes back once the cap lifts.
3. **On the same dots, MuJoCo's errors are 1.1–2.2× ours (about 2× on the deep presses)**, and larger on every press:

| Clip | Deepest | Ours | Our solver, MuJoCo's physics | MuJoCo | Dots moved (RMS) |
|---|---|---|---|---|---|
| T01 | 24 mm | 0.57 mm | 1.36 mm | 0.74 mm | 1.86 mm |
| T02 | 26 mm | 0.82 mm | 1.94 mm | 0.89 mm | 2.58 mm |
| T03 | 20 mm | 0.96 mm | 2.04 mm | 1.06 mm | 2.10 mm |
| T04 | 33 mm | 0.81 mm | 2.28 mm | 1.31 mm | 4.09 mm |
| T05 | 32 mm | 0.88 mm | 2.48 mm | 1.38 mm | 4.44 mm |
| T06 | 34 mm | 0.88 mm | 2.54 mm | 1.65 mm | 5.06 mm |
| T07 | 45 mm | 1.48 mm | 3.64 mm | 2.98 mm | 8.34 mm |
| T08 | 46 mm | 1.62 mm | 3.94 mm | 3.20 mm | 9.18 mm |
| T09 | 48 mm | 1.65 mm | 3.90 mm | 3.18 mm | 8.98 mm |
| R1 | 35 mm | 0.84 mm | 2.55 mm | 1.87 mm | 5.40 mm |
| R2 | 50 mm | 1.97 mm | 4.73 mm | 3.66 mm | 10.54 mm |
| H1 (held out) | 38 mm | 1.17 mm | 3.15 mm | 2.61 mm | 6.93 mm |
| O1 (held out) | 32 mm | 1.01 mm | 2.34 mm | 1.67 mm | 5.21 mm |

   "Our solver, MuJoCo's physics" does worse than MuJoCo at depth. It stays small-strain elastic
   throughout, while MuJoCo's collapsing top layer most likely happens to soak up some of the depth
   near the cap, a little as the crushed foam does. Neither is the right physics.
4. **Through time, it gets worse.** Run through H1 with the cap's measured motion (not just at fixed
   depths), MuJoCo's error over the whole press is 2.96 mm against our 1.17 mm. Its inverted elements
   also stay inverted after the cap lifts, so the top row of dots ends 8–9 mm low over the middle of the
   block, where the real foam's top row is back within 0.4 mm.
5. **Speed:** MuJoCo ran 85 to 130× slower than real time (8 to 12 minutes for H1's 5.7 s, depending on what else the
   machine was doing). Our model solves
   its depth table once (about 40 s) and then simulates any press in about a second.

`H1_ours_vs_mujoco.mp4` shows all three side by side: the real press, our block and MuJoCo's, given the
same cap motion. `H1_vs_mujoco_profiles.png` compares the rows of dots at the deepest point.

**Follow-up checks** test whether the gap comes from MuJoCo itself or from how it was set up.
`python scripts/mujoco_checks.py --trials trials.csv --out out` (MuJoCo 4.0, about 45 minutes on 2 cores)
writes `out/model/mujoco_checks.json` (the reference run's copy is in `results/model/`).

- **A finer grid doesn't stop the collapse.** With 13 vertex layers instead of 8, elements start turning
  inside out sooner (from 11 mm instead of 16 mm), which is what you'd expect if the material, not the
  grid, is the cause. On the pressing frames up to 29 mm the errors drop by 6–21% (H1: 0.93 to 0.85 mm).
- **MuJoCo's plasticity is the wrong kind.** Its J2 (von Mises) plasticity, with the yield strain set to
  our fitted 0.07, does make the force level off, but then the block stays crushed: after a 30 mm press
  and lift, 28 mm of the dent remains. Von Mises plasticity is metal-like flow at constant volume. Foam
  crushing loses volume and springs back; the real top row was back within 0.4 mm.
- **A material that can't turn inside out doesn't close the gap either.** MuJoCo 4.0 has an
  experimental Stable Neo-Hookean material, reachable only through its spec API (MJCF can't select it).
  With near-hard contact, so that the cap can't sink into the foam, the block follows the cap all the
  way to 56 mm and stays right side out to about 30 mm; a few elements invert from 37 mm on (8 layer
  pairs at 56 mm, where the default material has 6 by 16 mm). Poisson's ratio 0 fitted the training
  presses a shade better than 0.3 (2.73 against 2.74 mm). With the same contact for both materials, it
  predicts the dots worse than the default material on 9 of the 13 presses:

| Clip | Deepest | Ours | MuJoCo, default material | MuJoCo, Stable Neo-Hookean | Dots moved (RMS) |
|---|---|---|---|---|---|
| T01 | 24 mm | 0.83 mm | 1.15 mm | 2.23 mm | 2.78 mm |
| T02 | 26 mm | 1.22 mm | 1.38 mm | 3.08 mm | 3.94 mm |
| T03 | 20 mm | 1.12 mm | 1.34 mm | 2.59 mm | 2.24 mm |
| T04 | 33 mm | 1.10 mm | 1.82 mm | 2.79 mm | 5.71 mm |
| T05 | 32 mm | 1.07 mm | 1.68 mm | 2.92 mm | 5.48 mm |
| T06 | 34 mm | 1.15 mm | 2.03 mm | 2.91 mm | 6.66 mm |
| T07 | 45 mm | 1.82 mm | 4.08 mm | 4.05 mm | 10.69 mm |
| T08 | 46 mm | 1.99 mm | 4.41 mm | 4.35 mm | 11.59 mm |
| T09 | 48 mm | 2.33 mm | 5.07 mm | 4.80 mm | 13.13 mm |
| R1 | 35 mm | 1.07 mm | 2.24 mm | 3.08 mm | 7.17 mm |
| R2 | 50 mm | 2.42 mm | 4.99 mm | 4.50 mm | 13.18 mm |
| H1 (held out) | 38 mm | 1.33 mm | 2.78 mm | 3.31 mm | 8.22 mm |
| O1 (held out) | 32 mm | 1.17 mm | 2.08 mm | 2.21 mm | 6.11 mm |

  These are RMS errors over the 16 pressing frames per press that the shape was fitted on, so they
  differ from the table above, which covers every frame from contact to the end. The near-hard contact
  changes the default material's errors by less than 0.5 mm either way.

  Through time on H1 it scores 3.20 mm (default material 2.96 mm, ours 1.17 mm). Unlike the default
  material it leaves no dent: the top row is back at rest after the lift. It is also far too stiff once
  squashed. In a 30 mm press-and-lift it needs about 1.5 kN to push the cap 25 mm and 10 kN at 30 mm,
  where our model needs 135–160 N. (Even MuJoCo's default material needs about 120 N at 5 mm, against
  our 32 N, because it has no soft top layer and stands on a rigid table.)

So the gap is not the setup. A finer grid, MuJoCo's own plasticity and its inversion-proof material each
fix one symptom, and none brings the errors close to ours. What MuJoCo lacks is crushing: the cells'
buckling plateau, which concentrates the deformation under the cap, their stiffening as they close, and
the spring-back once the load comes off. MuJoCo remains the right tool for what it's built for, robots
with contact, and the fitted numbers carry over to it for shallow contact (under about 15% squash).
