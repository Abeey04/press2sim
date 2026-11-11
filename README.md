# press2sim

**Learning a foam block's stiffness and damping from video, then predicting presses it has never seen.**

A foam block with a painted grid of 105 dots is pressed by hand with a bottle and filmed in iPhone slow
motion. press2sim tracks how far each dot moves to a few hundredths of a millimetre and fits a 3D
finite-element model of the block: crushable foam in space, quasi-linear viscoelasticity in time. Given
only the bottle's depth over time, the fitted model then predicts every dot of a press it was never
fitted on.

![H1, a held-out press: the real clip with tracked and predicted dots, next to the simulated block](media/H1_demo.gif)

*H1, a press the model never saw. Left: the real clip, with the tracked dots in green and the predicted
dots in magenta. Right: the simulated block, driven only by the cap's measured depth.*
Full-quality videos: [H1](media/H1_simulation.mp4) · [O1, off-centre](media/O1_simulation.mp4) ·
[predicted dots on the original video](media/H1_model_vs_video.mp4) · [ours against MuJoCo](media/H1_ours_vs_mujoco.mp4)

**Try it in your browser:** [put a weight on the fitted block](https://abeey04.github.io/press2sim/explore.html),
drag it anywhere on the top and make it heavier or lighter. The page carries the model's 3D solutions and runs
its time behaviour live, so there is nothing to install.

## Results

| Held-out clip | What it is | Deepest | RMS error | RMS movement | Error / movement |
|---|---|---|---|---|---|
| H1 | a fresh press | 38 mm | 1.17 mm | 6.93 mm | 17% |
| O1 | off-centre, under the 5th dot column | 32 mm | 1.01 mm | 5.21 mm | 19% |
| F1 | a fingertip instead of the bottle | about 29 mm | 0.31 mm | 0.92 mm | 34% |

The error is the RMS over all 105 dots and every frame from contact to the end of the clip. On the 11
training presses it is 16–46% of the movement, and 16–20% on the deep ones. H1 and O1 are the clean tests.
F1 has no bottle to give its depth, so the fingertip's depth is read off the one dot under it through the
model, and that dot is left out of the score.

![O1, an off-centre press the model never saw, next to its simulation](media/O1_demo.gif)

*O1, pressed 30 mm left of the centre and also never used for fitting: the same model, with the cap where
it was.*

- **Stiffness:** E = 185 kPa at small strain. The cells start to buckle at 7% strain (a 12.9 kPa plateau)
  and the crushed cells touch from 55%. E is the least certain number: each dead weight alone gives
  101–190 kPa, and it leans on parts of the model the dots cannot pin down (see Limitations).
- **Damping:** 16% of the stiffness relaxes with a 0.56 s time constant. It is fitted to the one place it
  shows: the dent the cap leaves behind once the foam lets go of it. On the held-out presses that dent is
  predicted to 0.17 mm (H1) and 0.22 mm (O1), against 0.77 and 0.67 mm with no damping. Over a whole
  press, switching damping off changes the errors by about 0.01 mm.
- **MuJoCo baseline:** the same block in MuJoCo 4.0, scored the same way, has 1.1–2.2 times our error.
  A finer grid, MuJoCo's plasticity and its inversion-proof material do not close the gap
  ([details](docs/mujoco.md)).

![The real H1 press, our simulated block and MuJoCo's, all driven by the same measured cap depth](media/H1_vs_mujoco.gif)

*The same held-out press in our model (middle) and in MuJoCo (right), both given only the cap's measured
depth. MuJoCo's elements collapse under the cap and stay dented after the lift.*

## Check it yourself

Clone the repository, then (Python 3.10 or newer):

```bash
cd press2sim
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests          # 15 pass, 1 slow test is skipped
```

| Notebook | What it does | Needs | Time |
|---|---|---|---|
| [`02_check_results`](notebooks/02_check_results.ipynb) | Recomputes the held-out errors from the saved arrays, predicts H1 again from its cap depth alone, and shows where the error sits, what damping does, how uncertain E is, the MuJoCo comparison and what the soft top layer does | nothing beyond this repo | about 1 minute |
| [`01_reproduce`](notebooks/01_reproduce.ipynb) | Runs everything from the raw clips (frame timing, calibration, tracking, model, videos) and compares your numbers with `results/` | the 22 clips, 1.8 GB | about 1 hour on 2 cores |

**Raw clips:** [Google Drive folder](https://drive.google.com/drive/folders/1NeI0AOQUbnOajynEOH5SJd4AMVeiI9OX?usp=sharing)
(22 clips, 1.8 GB). They are 1080p HEVC at 240 fps nominal, with the audio removed and the video
untouched: every frame and timestamp is identical to what the camera recorded. Put them in `raw/`.
Every step is also a command, listed in [docs/pipeline.md](docs/pipeline.md).

The numbers under Limitations come from `python scripts/model_checks.py --trials trials.csv --out out`,
run after the model step. The reference run's copy is `results/model/model_checks.json`.

## How it works

1. **Capture:** one iPhone on a tripod films the block's dotted front face (15 × 7 dots) in slow motion.
   Slow motion drops frames, so every frame keeps its own timestamp.
2. **Calibration:** a printed checkerboard ([printable PDF](docs/checkerboard_260x110mm.pdf), 26 × 11
   squares of 10 mm) stands flat against the dotted face in the first clip, and its 250 inner corners map
   pixels to millimetres. Rulers on the foam then correct heights on the face by 4.9%, because the card
   sat slightly off the face's plane.

   <img src="media/calibration.jpg" width="760" alt="The printed checkerboard standing flat against the dotted face, and the same view without it">

   *Left: the printed checkerboard standing flat against the dotted face (calibration clip C1). Right: the
   same camera view without the card. The board's corners give the pixel-to-millimetre map for the dots.*
3. **Tracking:** each dot is a contrast-weighted centroid matched to the grid. Two dots on the bottle give
   the cap's depth and tilt. On synthetic clips with known truth, each dot's movement is accurate to
   0.03 mm and the cap's depth to 0.3 mm.
4. **Model:** a hexahedral finite-element grid (a voxel grid) of the whole 234 × 140 × 73 mm block. The
   foam is crushable (linear, a buckling plateau, densification), a thin top layer is much softer, the
   support gives a little, and time enters through Fung's quasi-linear viscoelasticity plus a dashpot.
   The dots ride on the face nodes.
5. **Fitting**, in three separate steps, so that each kind of observation sets only what it can: the
   shape from pressing frames, the damping from the dent the cap leaves behind, the stiffness from dead
   weights, the only clips with a known force. H1, O1 and F1 are never used. The shape search changes
   one parameter at a time from a hand-set start that already has the soft top layer
   ([docs/model.md](docs/model.md) says why and what happens without it).

Details: [docs/pipeline.md](docs/pipeline.md) (clips to tracked dots), [docs/model.md](docs/model.md)
(the model, the fit and every press's error), [docs/mujoco.md](docs/mujoco.md) (the baseline).

## Assumptions

- The foam is homogeneous and isotropic, so one set of parameters describes the whole block.
- Only motion in the face's plane is measured, so the model is checked in that plane alone.
- The bottle's cap is a flat, rigid 60 × 60 mm square, flush with the face, and its depth fully describes
  the load. The fingertip in F1 is an 8 mm-radius disc whose size and position were guessed.
- Small-strain elements with a secant crushable-foam law are used far past small strains: the presses
  reach 65% of the block's height, and the soft top layer under the cap is squeezed flat.

## Limitations

- **The soft top layer is a stand-in, not a measured layer.** At the deepest frame of every press, the
  dot 5 mm below the surface sits within a millimetre of the cap's bottom edge, so about 5 mm of the
  cap's travel never reaches the dots. The model absorbs it in a 3 mm top layer 18 times softer than
  the rest. Under the cap that layer is squeezed flat from about 11 mm of depth, and its elements end
  up slightly inside out (by under a millimetre). Broken cells at the cut surface are one possible
  cause; a rounded cap rim, the face's top edge folding over, or a few millimetres of bias in the cap's
  depth would look the same. Taking 3 mm off the cap's depth instead of using the layer gives a
  pressing-frame error of 1.71 mm, against 1.56 mm with the layer and 1.97 mm with neither. The dead
  weights do not support a soft layer under the whole pad: one E fits all four weights to 0.10 mm
  without it and to 0.23 mm with it.
- **The stiffness is uncertain by a factor of 2 to 3.** There was no force sensor. E comes from four dead
  weights through the model's soft top and its compliant support, and the dots pin down neither: E is
  185 kPa as fitted, 118 kPa without the soft top, 148 kPa on a rigid table and 71 kPa with neither. The
  predicted force to push the cap 30 mm in runs from 160 N down to 53 N accordingly. The spread between
  single weights (101–190 kPa) comes with the soft top, which starts to buckle under the pad at about
  2.4 kg in the model; the pad settling onto the block's uneven top may add to it. The dot motion in a
  press does not depend on E.
- **Damping rests on one small signal, and is only partly captured.** While the cap presses, its depth
  sets the dots, so damping shows only in the dent left once the foam lets go: under a millimetre, at
  six dots, for 1.3 to 4.4 s per clip. One relaxing fraction serves every press, yet presses deeper than
  about 32 mm keep a dent of 0.4–0.8 mm a second after release and the shallow ones keep none, so the
  fit improves the deep presses and makes T01 and T02 worse. A second, slower term helps a little (6% on
  the training presses), but these clips cannot set its time (2 to 10 s fit alike), so it is left out.
  The dashpot is not fitted at all: neither the dent nor the moment of release can tell 0 from 20 ms. And
  over all 105 dots around the lift, the fitted damping scores slightly worse than none (0.796 against
  0.790 mm), because the model's leftover dent is spread through the block where the real one stays
  under the cap.
- **Most of the error is at the deepest frames**, in the top row beside the cap and in the bottom row.
  Sideways motion of the top row is underpredicted: 6–7 mm against 9–12 mm measured in H1.
- **The camera setup adds small errors.** The checkerboard's hold-out error is 0.32 mm, the two
  checkerboard clips differ by 0.43 mm, and static features show 0.5–2 px of apparent camera shift within
  most clips. Each clip is measured from its own rest frame, and the tracks are not corrected for that
  shift.
- **One block, one indenter, presses by hand:** the fitted numbers describe this foam in this setup.

## What's in the repository

```
press2sim/     the package: video timing, calibration, dots and tracking, the 3D model, fitting, reports, MuJoCo baseline
notebooks/     01_reproduce (raw clips to every result) and 02_check_results (checks the results in a minute)
results/       the reference run behind every number in this README
media/         the demo GIF and videos
docs/          details on the pipeline, the model and the MuJoCo comparison, and explore.html (the interactive page)
scripts/       synthetic test clips with known truth, the checks behind the limitations, and the MuJoCo follow-up checks
tests/         unit tests and checks on real frames
trials.csv     which clip is which, with the dead-weight readings
```

## Credits

Code and tools this project uses:

- [NumPy](https://numpy.org) and [SciPy](https://scipy.org): arrays, sparse matrices, image filters, the finite-element assembly and solves.
- [OpenCV](https://opencv.org): checkerboard corners (`findChessboardCornersSB`), the homography, and the optical flow behind the camera-still check.
- [PyAV](https://pyav.org) on [FFmpeg](https://ffmpeg.org): decoding the iPhone clips with each frame's own timestamp. FFmpeg also encodes the output videos when it is installed.
- [matplotlib](https://matplotlib.org): every plot and the drawn frames of the videos.
- [PyAMG](https://github.com/pyamg/pyamg): the algebraic multigrid preconditioner that makes the 3D block solves about ten times faster.
- [MuJoCo](https://mujoco.org) 4.0 (Google DeepMind): the baseline comparison, including its experimental Stable Neo-Hookean material.
- [pandas](https://pandas.pydata.org) and [Jupyter](https://jupyter.org): the notebooks.
- [pytest](https://pytest.org): the tests.
- [three.js](https://threejs.org) r128, loaded from cdnjs: the interactive page.
- Optional: [CoTracker3](https://co-tracker.github.io) (Meta AI, CC-BY-NC) for the tracking cross-check, loaded through PyTorch Hub.

The physical model follows standard references: the linear, plateau and densification regimes of crushable
foam (Gibson and Ashby, *Cellular Solids*) and Fung's quasi-linear viscoelasticity (Fung, *Biomechanics*).
