"""End to end on rendered clips with known truth. Slow (several minutes): set PRESS2SIM_SLOW=1 to run."""
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
                    "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out"), "--pairs", ""], check=True, cwd=ROOT)
    T = np.load(tmp_path / "truth.npz")
    cal = json.loads((tmp_path / "out" / "calib" / "C1.json").read_text())
    assert cal["holdout_rms_mm"] < 0.15
    for code in ("S1", "T05"):
        A = np.load(tmp_path / "out" / "tracks" / f"{code}.npz")
        k0 = int(A["k0"])
        assert np.allclose(A["t"], T[f"{code}_t"] - T[f"{code}_t"][0], atol=1e-6)
        err = A["pts_mm"][k0:, :105] - T[f"{code}_dots"][k0:]
        assert np.isfinite(err).mean() > 0.99
        assert np.sqrt(np.nanmean(err ** 2)) < 0.1
    A = np.load(tmp_path / "out" / "tracks" / "T05.npz")
    ok = np.isfinite(A["cap_z_mm"])
    assert np.sqrt(np.mean((A["cap_z_mm"][ok] - T["T05_cap_bottom_z"][ok]) ** 2)) < 0.1
    ev = json.loads((tmp_path / "out" / "tracks" / "T05.json").read_text())["events"]
    assert abs(ev["max_depth_mm"] - 29.0) < 0.2
    assert abs(ev["t_contact"] - 1.5375) < 0.01
