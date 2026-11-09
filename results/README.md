# Reference results

The run behind every number in the main README, made with `notebooks/01_reproduce.ipynb` (or the same
commands by hand) on the session's 22 clips. `notebooks/02_check_results.ipynb` reads these files, and
`01_reproduce.ipynb` compares a fresh run against them. A fresh run writes the same files to `out/`.

| Folder | What is in it |
|---|---|
| `calib/` | `C1.json` and `C2.json`: the checkerboard calibrations (homography, hold-out error, face-plane correction); `compare.json`: C1 against C2 |
| `tracking/` | `summary.csv`: one row per clip (frame rate, rest noise, coverage, camera shift, depth, speed); `report.json`: noise floor and repeat presses |
| `model/` | `params.json`: the fitted parameters and every fit's history; `predictions.csv`: the error for every press; `stiffness_per_weight.json`: E fitted to each dead weight alone; `start.json`: the fitted shape that `--keep-shape` reuses; `mujoco_comparison.*` and `mujoco_checks.json`: the MuJoCo baseline |
| `tracks/` | the tracked dots of the three held-out clips, H1, O1 and F1 (`.npz` arrays, `.json` facts; see `docs/pipeline.md`) |
| `predictions/` | the model's predicted dot movement for H1, O1 and F1, with and without damping, and MuJoCo's for H1 |
| `figures/` | calibration, tracking and model plots for the held-out presses and T09 |

The tracks and predictions of the training clips are not stored here, to keep the repository small;
`01_reproduce.ipynb` regenerates them.
