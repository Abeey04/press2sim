"""End to end on rendered clips with known truth. Slow (several minutes): set PRESS2SIM_SLOW=1 to run.

The tight checks are on what the model is fitted to: how far each dot moves from its rest position, and
the cap's depth. Positions themselves are only held to 0.5 mm, because x is measured from the block's
left end and heights from its base, and both come from the foam's outline, which is good to a few
tenths of a millimetre. The rendered press ends with an instantaneous release that no hand makes; the
bottle is lost there, so the lift and release events are not checked.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(not os.environ.get("PRESS2SIM_SLOW"), reason="set PRESS2SIM_SLOW=1")
def test_pipeline_matches_truth(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), str(tmp_path), "--rotate180"], check=True)
    subprocess.run([sys.executable, "-m", "press2sim", "all", "--trials", str(tmp_path / "trials.csv"),
                    "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out"), "--pairs", "",
                    "--config", str(tmp_path / "config.json")], check=True, cwd=ROOT)
    T = np.load(tmp_path / "truth.npz")
    cal = json.loads((tmp_path / "out" / "calib" / "C1.json").read_text())
    assert cal["holdout_rms_mm"] < 0.15
    for code in ("S1", "T05"):
        A = np.load(tmp_path / "out" / "tracks" / f"{code}.npz")
        k0 = int(A["k0"])
        truth = T[f"{code}_dots"]
        assert np.allclose(A["t"], T[f"{code}_t"] - T[f"{code}_t"][0], atol=1e-6)
        position = A["pts_mm"][k0:, :105] - truth[k0:]
        assert np.isfinite(position).mean() > 0.99
        assert np.sqrt(np.nanmean(position ** 2)) < 0.5
        movement = A["disp_mm"][k0:] - (truth[k0:] - truth[k0])        # the rendered foam is still at k0
        assert np.sqrt(np.nanmean(movement ** 2)) < 0.06
    A = np.load(tmp_path / "out" / "tracks" / "T05.npz")
    seen = np.isfinite(A["cap_z_mm"])
    assert seen.sum() > 300                                             # the whole press and hold
    depth = float(A["foam_top_mm"]) - A["cap_z_mm"][seen]
    assert np.sqrt(np.mean((depth - (73.0 - T["T05_cap_bottom_z"][seen])) ** 2)) < 0.5
    ev = json.loads((tmp_path / "out" / "tracks" / "T05.json").read_text())["events"]
    assert abs(ev["max_depth_mm"] - 29.0) < 0.5
    assert abs(ev["t_contact"] - 1.5375) < 0.02
    assert abs(ev["hold_s"] - 1.0) < 0.05
