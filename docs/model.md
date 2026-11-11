# The model: what it is, how it was fitted, what it predicts

`python -m press2sim model --trials trials.csv --raw raw` fits the model on the training clips,
predicts every press (training and held out), and writes to `out/model/`:

- `params.json`: the fitted parameters, effective stiffness and damping, every fit's history
- `predictions.csv`: the error for every clip
- `H1_*.png`, `O1_*.png`, `F1_*.png`: model against measurement for the held-out clips
- `H1_model_vs_video.mp4`, `O1_model_vs_video.mp4`: the prediction drawn on the original video
- `H1_3d.png`: the model block at the deepest point
- `H1_simulation.mp4`, `O1_simulation.mp4`: the simulated 3D block next to the real clip, both driven by the
  same cap motion (made by `python -m press2sim simulate --trials trials.csv --raw raw`)
- `pred_<code>.npz`: the predicted dot movement for every press

Add `--keep-shape` to reuse the shape from `out/model/start.json` and refit only the damping and stiffness (a few minutes instead of 20).

`python scripts/model_checks.py --trials trials.csv --out out` then reruns the checks quoted under
"How to read this" and "Limits" below, and writes them to `out/model/model_checks.json` (about 15
minutes; `--part numerics`, the finer grids, takes another 40).

The reference run's copies are in `results/model/` and `results/predictions/`; every number below comes from them.

## What it is

A 3D model of the block, driven only by what pushes on it.

- **Space:** a voxel grid of the whole block. The 234 × 139.5 × 73 mm block is cut into
  hexahedral elements (7.5 mm near the dotted face and the top, coarser at the back). Each element
  is a crushable foam: linear up to a strain ε_y, then a soft plateau where the cells buckle (slope
  h·E), then stiffening again once the crushed cells touch (from ε_d). The top 3 mm is a much
  softer layer, a stand-in for whatever keeps the first few millimetres of the cap's travel from
  reaching the dots (see Limits). The cap is a flat rigid
  60 × 60 mm square pushed into the top, its front edge level with the dotted face. The block stands
  on a slightly compliant support. The dead-weight clips showed this clearly: under the 2.4 kg
  weight the bottom row of dots sank about half as far as the top row, where a block on a rigid
  table would barely move at the bottom. The dots are carried by the grid like particles in a particle-voxel scheme:
  each one moves with the face nodes around it.
- **Time:** quasi-linear viscoelasticity (Fung), plus a dashpot. The force on the foam is
  `F(t) = E [g∞ R(q) + h + τ_v dR(q)/dt]`, with `dh/dt = g dR(q)/dt − h/τ` and `g∞ = 1 − g`. R(q)
  is the grid's elastic force at indentation q. The relaxing part h makes the force fall while the
  cap holds still; the dashpot τ_v limits how fast the foam can spring back. While the cap touches
  the foam, q is the cap's measured depth. When the foam would have to pull on the cap to keep up
  (F < 0), the two part and q follows F = 0: the foam rises on its own. Under a dead weight,
  F = m g instead.

**Effective stiffness** is E (the small-strain Young's modulus) together with the plateau; the
force on the cap at any depth follows from the grid. **Effective damping** is the relaxing fraction g
with its time constant τ (a viscosity of g·E·τ), and the dashpot τ_v.

## How it was fitted, in three separate steps

The clips are split: training presses fit the model; H1 (a fresh press), O1 (off-centre, under the
5th column) and F1 (a fingertip) are never used for fitting.

1. **Shape** (ε_y, h, ε_d, Poisson's ratio ν, the soft top layer, the cap's position relative to
   the face): 16 frames from each training press, spread over the depths it went through while
   the cap was pushing. With the cap's depth imposed, quasi-linear viscoelasticity leaves the
   deformation's shape purely elastic, so these can be fitted before the damping. A coordinate
   search over the parameters, about 20 grid solves. The support comes from the pattern of the
   dead-weight clips (how far the bottom row sinks compared with the top row), before and after.

   The search starts from a hand-set point, `SHAPE_START` in `press2sim/fit.py`: a soft top of 3 mm
   at 0.1 × E, ε_y = 0.06, h = 0.01, ε_d = 0.64, ν = 0, and the cap's front edge 3 mm in front of the
   face. These values came from earlier exploratory fits. The start has the soft top on because the
   search changes one parameter at a time, and the layer only exists once it has both a thickness
   and a softness. (Given a start with no layer, the search now tries the two together first.) A
   fit with no soft top at all stops at a pressing-frame error of 1.92 mm instead of 1.56 mm, and
   predicts H1 to 1.48 mm and O1 to 1.19 mm instead of 1.17 and 1.01 mm. Every step the search
   tried, with its score, is in `results/model/params.json` under `shape` → `history`.
2. **Damping** (g and τ): the dent the cap leaves behind. While the cap presses, its depth sets the
   dots and damping changes nothing. Once the foam lets go of the cap, how fast the dent closes is
   all damping. So the fit scores the vertical movement of the six top-row dots under the cap, from
   the measured release to the end of the clip (1.3 to 4.4 s later). The dashpot τ_v is not fitted
   (see "How to read this").
3. **Stiffness E**: the dead-weight clips K1–K4, where the force is known (m g). Hand presses can't
   give E, because the hand's force was never measured. In return the model *predicts* the force the
   hand applied in every press.

## The demo

For a held-out press the model gets the cap's measured depth over time (from the two bottle dots)
and nothing else. It predicts all 105 dots at every frame, which are then compared with the tracked
dots.

![The model's predicted dots drawn on the original H1 video, next to the tracked dots](../media/H1_overlay.gif)

*H1, played at about 0.6 times real speed: tracked dots in green, the model's prediction in magenta.*

## Fitted parameters

| What | Value | From |
|---|---|---|
| Young's modulus at small strain, E | 185 kPa | dead weights K1–K4 |
| Cells start to buckle at strain ε_y | 0.070 (plateau stress 12.9 kPa) | presses |
| Plateau slope h | 0.070 × E | presses |
| Crushed cells touch from strain ε_d | 0.55 | presses |
| Poisson's ratio ν | 0.00 | presses |
| Soft top layer (a stand-in, see Limits) | 3.0 mm at 0.055 × E | presses |
| Cap's front edge relative to the dotted face | flush (0 mm) | presses |
| Support under the block | 0.0296 × E per mm | dead weights |
| Damping: fraction of the stiffness that relaxes, and how fast | 0.16, τ = 0.56 s (viscosity 16.1 kPa·s) | the dent under the cap after release |
| Damping: dashpot in parallel (limits how fast the foam springs back) | τ_v = 1.24 ms, held and not fitted | these clips cannot set it (see below) |

Effective stiffness of the block under the cap (fast = before any relaxation, long-term = after it):

| Depth | Force, fast | Force, long-term | Stiffness, fast | Stiffness, long-term |
|---|---|---|---|---|
| 5 mm | 32.1 N | 27.1 N | 11.24 N/mm | 9.49 N/mm |
| 10 mm | 71.7 N | 60.6 N | 5.71 N/mm | 4.82 N/mm |
| 20 mm | 111.2 N | 94.0 N | 4.08 N/mm | 3.44 N/mm |
| 30 mm | 159.7 N | 134.9 N | 4.93 N/mm | 4.17 N/mm |

Under the full pad the block's stiffness is 38 N/mm.

## Predictions

RMS error of all 105 dots over every frame from just before contact to the end of the clip, against the RMS movement of the dots over the same frames. The model gets only the cap's depth over time. The last column is the error in the dent the cap leaves behind: the vertical movement of the top-row dots under the cap, from the measured release to the end of the clip, with the same error without damping in brackets.

| Clip | Set | Deepest | RMS error | RMS movement | Error / movement | At the deepest frame | Top / middle / bottom row | No damping | Dent after release (no damping) |
|---|---|---|---|---|---|---|---|---|---|
| T01 | training | 23.9 mm | 0.57 mm | 1.86 mm | 30% | 1.20 mm | 0.73 / 0.62 / 0.30 mm | 0.56 mm | 0.33 mm (0.33 mm) |
| T02 | training | 25.6 mm | 0.82 mm | 2.58 mm | 32% | 1.34 mm | 0.95 / 1.00 / 0.43 mm | 0.81 mm | 0.41 mm (0.30 mm) |
| T03 | training | 20.4 mm | 0.96 mm | 2.10 mm | 46% | 1.55 mm | 1.06 / 1.10 / 0.55 mm | 0.96 mm | 0.84 mm (0.88 mm) |
| T04 | training | 32.6 mm | 0.81 mm | 4.09 mm | 20% | 1.29 mm | 1.19 / 0.74 / 0.35 mm | 0.80 mm | 0.38 mm (0.40 mm) |
| T05 | training | 31.9 mm | 0.88 mm | 4.44 mm | 20% | 1.23 mm | 1.24 / 0.83 / 0.38 mm | 0.88 mm | 0.20 mm (0.61 mm) |
| T06 | training | 34.0 mm | 0.88 mm | 5.06 mm | 17% | 1.34 mm | 1.32 / 0.74 / 0.33 mm | 0.89 mm | 0.17 mm (0.68 mm) |
| T07 | training | 45.1 mm | 1.48 mm | 8.34 mm | 18% | 2.31 mm | 2.06 / 1.12 / 0.79 mm | 1.47 mm | 0.19 mm (0.75 mm) |
| T08 | training | 45.5 mm | 1.62 mm | 9.18 mm | 18% | 2.28 mm | 2.26 / 1.28 / 0.82 mm | 1.62 mm | 0.23 mm (0.90 mm) |
| T09 | training | 47.6 mm | 1.65 mm | 8.98 mm | 18% | 2.65 mm | 2.27 / 1.33 / 0.81 mm | 1.65 mm | 0.40 mm (0.76 mm) |
| R1 | training | 35.0 mm | 0.84 mm | 5.41 mm | 16% | 1.22 mm | 1.29 / 0.65 / 0.44 mm | 0.84 mm | 0.42 mm (0.78 mm) |
| R2 | training | 50.0 mm | 1.97 mm | 10.54 mm | 19% | 2.89 mm | 2.78 / 1.60 / 0.95 mm | 1.97 mm | 0.16 mm (0.93 mm) |
| H1 | held out | 38.4 mm | 1.17 mm | 6.93 mm | 17% | 1.57 mm | 1.72 / 0.83 / 0.62 mm | 1.17 mm | 0.17 mm (0.77 mm) |
| O1 | held out | 32.4 mm | 1.01 mm | 5.21 mm | 19% | 1.56 mm | 1.64 / 0.79 / 0.59 mm | 1.01 mm | 0.22 mm (0.67 mm) |
| F1 | held out: fingertip, driven by the one dot under it | 28.9 mm (inferred) | 0.31 mm | 0.92 mm | 34% | 0.58 mm | 0.39 / 0.26 / 0.12 mm | 0.31 mm | not scored |


How to read this:

- **The shape carries the prediction.** Deep presses are predicted to within 16–20% of how far the
  dots moved, and the held-out presses no worse than the training ones: H1 (a fresh press, 38 mm)
  17%, O1 (off-centre, a different FE run with the cap 30 mm to the left) 19%. The shallow presses
  T01–T03 have larger relative errors because the dots moved less, not because the errors are
  larger. The crushable foam matters: the deformation spreads deep at first, concentrates near the
  cap once the cells start buckling, and spreads deep again as the crushed layer stiffens. A
  linear elastic block spreads it the same way at every depth.
- **Damping shows in the dent the cap leaves behind, and nowhere else.** Over a whole press the
  no-damping column is the same to about 0.01 mm: while the cap presses, its depth sets the dots.
  After the foam lets go, the dent under the cap is predicted to 0.46 mm RMS on the training
  presses and 0.20 mm on H1 and O1, against 0.71 and 0.72 mm with no damping. One second after
  release that dent is 0.41 mm deep in H1 and 0.44 mm in O1; the model has 0.52 and 0.44 mm, and
  none at all without damping. The fit is well set: started from 12% at 0.34 s or from 30% at
  2 s, it lands on the same 16% and 0.56 s. What it does not capture:
  - *One relaxing fraction for every press.* Presses deeper than about 32 mm keep a dent of 0.4 to
    0.8 mm a second after release (T04, at 33 mm with no hold, keeps 0.1 mm). The shallow T01 and
    T02 (24 and 26 mm) keep none and end 0.15 to 0.2 mm above where they started: at rest the
    bottle stood on the foam, and after the press it is gone. The fit trades them off. The deeper
    presses improve, T01 and T02 get worse (T02 from 0.30 to 0.41 mm).
  - *Where the dent sits.* The model's leftover dent has the shape of a shallow press, spread
    through the block; the real one stays under the cap. Scored over all 105 dots around the lift,
    the fitted damping is slightly worse than none (0.796 against 0.790 mm).
  - *A slower part.* A second relaxing term lowers the training score by 6%, to 0.43 mm (fractions
    18% and 5%, times 0.28 and 4.5 s), and the held-out score to 0.16 mm. But its time is not set
    by these clips: held at 2, 3, 5 or 10 s it scores within 1%, because 9 of the 11 records end
    less than 2.5 s after release. So the model keeps one term.
  - *The dashpot.* It is not fitted. With g and τ as fitted, the dent score stays within 1% for
    any dashpot from 0 to 20 ms, and the moment the model's foam lets go of the cap does not move
    at all over that range (21 ms late, median over the training presses). A fitted value would
    only add a viscous force while the cap presses fast, which nothing here measures. It is held
    at 1.24 ms, the value an earlier fit to whole presses returned.
  - *T03.* Its "lift" is the hand easing off by 2.6 mm for two seconds before the real lift, and
    the dots under the cap stayed down. No damping setting changes its score much (0.84 to
    0.88 mm).
- **E is the least certain number, by a factor of 2 to 3.** Fitted to each weight alone it comes out
  as K1 101, K2 145, K3 170, K4 190 kPa: the heavier the weight, the stiffer the block looks. That
  spread comes with the soft top, which in the model starts to buckle under the pad at about 2.4 kg;
  the pad settling onto the block's uneven top (it dips about 2 mm in the middle) may add to it.
  More important, E depends on two parts of the model that the dots do not pin down. Refitting E
  to the same weights gives 185 kPa as fitted, 118 kPa without the soft top (and then one E fits
  all four weights to 0.10 mm instead of 0.23 mm), 148 kPa on a rigid table, and 71 kPa with
  neither. The force to push the cap 30 mm in goes from 160 N to 53 N accordingly. The dots'
  movement in a press doesn't depend on E at all; only the predicted forces do.
- **The fingertip (F1)** is a rough stand-in: an 8 mm-radius disc whose centre is 9 mm behind the
  face, both guessed rather than measured, driven by the one dot under it. It still predicts the
  other 104 dots to 0.31 mm (34% of their movement).

## Limits

- **The soft top layer is a stand-in.** At the deepest frame of every press the dot 5 mm below the
  surface sits within a millimetre of the cap's bottom edge (the cap's depth minus that dot's sink is
  4–6 mm in all 13 presses, whether 20 or 50 mm deep). About 5 mm of the cap's travel never reaches
  the dots, and the model absorbs it in the 3 mm layer at 0.055 × E. Under the cap that layer is
  1.9 mm thick at 1.5 mm of depth, 0.2 mm at 7 mm, and from 11 mm on it is slightly inside out (by
  0.1 to 0.8 mm; 66 to 90 of the 5,280 elements pass 100% compression). Small-strain elements have
  nothing to stop that. The cause is open: broken cells at the cut surface, a rounded cap rim, the
  face's top edge folding over, or a bias of a few millimetres in the cap's depth would all look the
  same in the dots. With no layer and 3 mm taken off the cap's depth instead, the pressing-frame
  error is 1.71 mm (1.56 mm with the layer, 1.97 mm with neither, other parameters unchanged). The
  dead weights speak against a soft layer under the whole pad (see above).
- **Numerical checks.** Refining the grid from 7.5 mm to 3.75 mm changes H1's error by 0.013 mm and
  the forces by under 3%. At 29 mm and deeper, the element-stiffness iteration stops at its 10-pass
  limit with a few elements not settled; running 80 passes changes the forces by 2% or less and H1's
  error by under 0.001 mm.
- Small-strain elements at up to 65% compression. Most of the remaining error is at the deepest
  frames of the deepest presses: the top row beside the cap, and the bottom row. In T09 the bottom
  row sinks about 6 mm along the whole length of the block; the model gets that under the cap but
  only about half of it at the ends.
- The face's movement towards or away from the camera is invisible, so the model is only checked
  in the face's plane.
- Sideways movement of the top row towards the cap is underpredicted: in H1 about 6–7 mm against
  9–12 mm measured.
- One block, one indenter shape, presses by hand: the fitted numbers describe this foam in this
  setup.
