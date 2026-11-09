# From clips to tracked dots

Everything up to the tracked dots: what to run, what comes out, how it works and how it was validated.
The model is in [model.md](model.md) and the MuJoCo baseline in [mujoco.md](mujoco.md).

## Put the clips in place

1. Put the original `.MOV` files in a folder named `raw` (for this session: the download link in the
   README). If you record your own, copy them over USB with *Keep Originals* on and never re-save or
   re-export them: that rewrites the frame timestamps.
2. `trials.csv` says which file is which clip, with the scale readings for the dead weights K1–K4. The
   one in the repo is filled in for this session. For your own session, the sticky-note slate at the
   start of each clip tells you which is which.

## The calibration board

Print `docs/checkerboard_260x110mm.pdf` at 100% (actual size, no fit-to-page) and check its 100 mm scale
bar with a ruler: the checker area must measure 260 × 110 mm (26 × 11 squares of 10 mm, so 25 × 10
inner corners). Glue it to stiff card, cut along the dashed line, and film it standing flat against
the dotted face with its bottom edge on the table: that is clip C1. Film it again at the end of the
session (C2) to check that nothing moved. If the print came out at another size, put the measured size
in `board.printed_width_mm` and `board.printed_height_mm` (see `config.example.json`) and pass that file
with `--config`. `media/calibration.jpg` shows the board in this session, and
`results/figures/calib_C1.png` the corners it found with their residuals.

## Run, in this order

| Step | Command | What to look for |
|---|---|---|
| 1. Frame timing | `python -m press2sim times raw` | Average around 195–240 fps with some "long gaps": that's the iPhone dropping frames. It's harmless because each frame's real timestamp is used. |
| 2. Calibration | `python -m press2sim calib --trials trials.csv` | Squares measuring 10.0 mm and a hold-out error of a few tenths of a millimetre (0.32 mm in this session; the tool warns above 0.2 mm). C1 against C2 differed by 0.43 mm here, so the line ends in `check`. Then open `out/qa/calib_C1.png`. |
| 3. Two clips first | `python -m press2sim track --trials trials.csv --only S1,T05` | Open `out/qa/T05_frames.png`, `T05_timeseries.png` and `T05_field.png`. |
| 4. Everything | `python -m press2sim track --trials trials.csv --jobs 4` | One line per clip; see the checklist below. |
| 5. Report | `python -m press2sim report --trials trials.csv` | Noise floor from S1, repeatability from R1/T05 and R2/T08, and `out/summary.csv`. |
| 6. Model | `python -m press2sim model --trials trials.csv --raw raw` | About 20 minutes on 2 cores. Writes `out/model/`: the fitted parameters, the prediction errors, plots and the comparison videos. See [model.md](model.md). |
| 7. Videos | `python -m press2sim simulate --trials trials.csv --raw raw` | `out/model/H1_simulation.mp4` and `O1_simulation.mp4`: the simulated block next to the real clip. |
| 8. MuJoCo (optional) | `python -m press2sim mujoco --trials trials.csv --raw raw`, then `python scripts/mujoco_checks.py --trials trials.csv --out out` | See [mujoco.md](mujoco.md). |

`notebooks/01_reproduce.ipynb` runs all of these in order and compares your numbers with `results/`.

Each line from step 4 reports the following. Check them:

- **Rest frame:** found shortly after the slate, with **105/105 dots**.
- **Tracked percentage:** close to 100%. A few lost frames right under the cap in the 29 mm presses are fine; look at the clip's `_frames.png`.
- **Rest noise:** 0.01–0.09 mm in this session. S1, where nothing touches the foam, gives the floor: a median
  standard deviation of 0.010–0.015 mm per dot.
- **Camera shift:** the apparent shift of static features on the table and at the frame's edges. It was
  0.5–2.2 px in this session (0.1–0.6 mm), and 7.5 px in K0, where the pad went on; ideally it stays
  under about 0.3 px. The tracks are not corrected for it.
- **Bottle dots:** about **46.5 mm apart** (55 − 8.5). Anything else means the wrong dots were picked.
- **Tilt:** a few degrees at most.

## What comes out

`out/tracks/<code>.npz` holds the arrays:

| Key | Shape | Meaning |
|---|---|---|
| `t` | (T,) | seconds from the clip's first frame, from the file's own timestamps |
| `pts_mm` | (T, N, 2) | x, z of every point in mm. x is measured from the block's left end along the front face; z is height above the table |
| `disp_mm` | (T, 105, 2) | movement of each dot from its rest position |
| `rest_mm` | (105, 2) | rest positions, averaged over the still frames before the press |
| `valid` | (T, N) | point tracked in this frame; otherwise NaN in `pts_mm` |
| `pts_px` | (T, N, 2) | the same points in image pixels |
| `cap_x_mm`, `cap_z_mm` | (T,) | press clips: centre of the cap's flat bottom face, from the two bottle dots |
| `tilt_deg` | (T,) | press clips: bottle tilt in the face plane, positive when the top leans right |
| `k0` | int | index of the rest frame where tracking starts |

- **Dot order:** index = row × 15 + column. Row 0 is the bottom row (z ≈ 8 mm) and column 0 the leftmost (x ≈ 47 mm).
- **Extra points in press clips:** index 105 is the cap dot and 106 the tape dot.
- **Gaps:** don't interpolate across long runs of NaN.

`out/tracks/<code>.json` holds that clip's facts:
- frame timing and the rest frame
- grid and foam-outline measurements
- the events: contact, deepest point, lift and release, plus the maximum depth, the press speed and the hold time
- tracking coverage and the camera-still check

## How it works

- **Time:** iPhone "240 fps" slow motion drops frames, so every frame keeps the presentation timestamp stored in the file (read with PyAV). Frame index divided by a nominal rate would be wrong.
- **Calibration:** the checkerboard is found in every still frame of C1 and its 250 inner corners are averaged. A homography then maps image pixels to the card plane. It's checked by fitting on half the corners and testing on the other half. C1 against C2 says whether the camera or the foam moved, ignoring where the card happened to be placed sideways.
- **Face-plane check:** the card is only a proxy for the dotted face, so the face is measured against rulers on the foam itself, in the frames of C1 before the card goes in. In this session the printed board was exactly to scale (260 × 110 mm), but on the face heights read about 5% short: the dot rows, 59 mm apart by ruler, came out 56.5 mm, and the block's 73 mm height came out 69.3 mm. Widths agreed (the 234 mm length read 235.7 mm). So the card was not quite in the face's plane. Heights on the face are rescaled by the average of the two height references (×1.049; they agree within 0.9%), with z = 0 at the foam's base. The measurement and both references are recorded in `out/calib/C1.json` under `face`. Put your own ruler reading in `grid.measured_row_span_mm` in `config.py`.
- **Rest frame:** the first stretch where all 105 dots are visible and still. This skips the slate and your hand.
- **Dots:** a morphological closing estimates the local background. Unlike a maximum filter, it doesn't smear the foam's bright sparkles. Each dot's centre is the contrast-weighted centroid of its blob, accurate to a small fraction of a pixel. The blobs are matched to a 15 × 7 lattice that allows for hand-marking errors.
- **Tracking:** every frame, all dark blobs are found at once, and each dot takes the blob nearest its predicted position. A blob that is the wrong size or too faint is marked invalid instead of guessed; dots that lose their blob get a local second look.
- **Bottle:** at rest, the cap dot is the lowest dark mark on the bottle that sits on yellow, with the tape dot about 46.5 mm above it. Each frame, the two are found near their predicted positions, each using its own contrast (the brown-on-yellow cap dot is much fainter than the foam dots). Foam dots and the other bottle dot are masked out of the search, and the two must stay the bottle's length apart. The two dots give the bottle's axis; the cap's flat bottom is 8.5 mm below the cap dot along it. The bottle's front face is not in the face's plane, so the bottle's own scale comes from the dots' separation at rest. If one dot is hidden for a moment, the other one plus the slowly changing offset between them stands in for it.
- **Depth:** measured from the undisturbed top surface beside the cap. The block's top is not level (it dips about 2 mm in the middle), and the bottle resting on it at rest already presses it down about half a millimetre.
- **Camera still:** static features on the table in front of the foam and at the far sides of the frame are followed with optical flow.

## Validation

- **Synthetic clips with known truth** (`scripts/make_synthetic.py`). These include a slightly turned camera, about 18% dropped frames, HEVC compression, storage rotated 180° with a display tag, a slate in front of the dots, 100 Hz flicker and sensor noise. Results:

  | Check | Result |
  |---|---|
  | Calibration hold-out error | 0.06 mm |
  | Dot positions | 0.015 mm RMS, worst 0.06 mm, during a 29 mm press |
  | Cap height | 0.014 mm RMS |
  | Bottle tilt | within 0.03° |
  | Contact time | within one frame |
  | Maximum depth | 29.0 mm |
  | Frames tracked | 100% |

  To rerun it (about 6 minutes): `python scripts/make_synthetic.py syn --rotate180`, then `python -m press2sim all --trials syn/trials.csv --raw syn/raw --out syn/out --pairs ""`. Or set `PRESS2SIM_SLOW=1` and run pytest.
- **Real frames** in `tests/data`, from the session: all 105 dots found and labelled, and both bottle dots found.
- **The real session**, for comparison: calibration hold-out error 0.32 mm, rest noise 0.01–0.09 mm per clip,
  and an apparent camera shift of 0.5–2.2 px within most clips. The README lists these under limitations.

## Optional: CoTracker3 cross-check (GPU)

Install PyTorch with CUDA from pytorch.org, then run:

```
python -m press2sim.cotracker_check --code T05
```

It tracks the same dots with the pretrained CoTracker3 online model and reports how far it disagrees with the centroid tracker, in mm. A large disagreement on a dot points to a tracking problem to look at. CoTracker is licensed CC-BY-NC; credit it if you report its numbers.

## Settings and troubleshooting

`press2sim/config.py` lists every physical number: block size, dot grid, board, and the cap-dot and tape-dot heights. To override any of them without editing code, pass a JSON file with `--config`. For example, if the printed board measured 259.5 mm wide:

```json
{"board": {"printed_width_mm": 259.5}}
```

- **"never saw all 105 dots still"**: something stayed in front of the face. If the dots are faint, lower `tuning.contrast_fraction`.
- **Video upside down:** add `--rotate 180`. Rotation is normally read from the file.
- **Checkerboard not found:** the whole board must be in view, flat and sharp.
- **`--jobs` misbehaves on your machine:** use `--jobs 1`.
