"""export_pose_tokens.py on REAL inputs: the two EE4D-Motion pose exports in ~/Downloads/expertaf_test_poses
and a real (smoke-trained, 2 takes x 40 epochs) tokenizer checkpoint from train_pct_tokenizer.py there.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_export_pose_tokens.py -q
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import export_pose_tokens as ep  # noqa: E402
import train_pct_tokenizer as tp  # noqa: E402

POSES = os.path.expanduser("~/Downloads/expertaf_test_poses")
WEIGHTS = os.path.join(POSES, "pct_smoke_best_avg_joint_loss_epoch_23.pth")
TAKES = ["iiith_soccer_004_2", "sfu_basketball013_11"]


def test_procrustes_is_expertafs_and_removes_similarity_transforms():
    pa = ep.expertaf_procrustes()
    x = torch.randn(4, 17, 3)
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.det(R) < 0:
        R[:, 0] *= -1
    y = 1.7 * x @ R.T + torch.tensor([3.0, -2.0, 0.5])
    assert torch.allclose(pa(x, y), torch.zeros(4), atol=1e-4)          # scale+rotation+shift: zero error
    assert pa(x, x + 0.01).mean() > 0                                   # a real difference remains


def test_export_writes_expertaf_token_files(tmp_path):
    summ = ep.main(["--pose-dir", POSES, "--weights", WEIGHTS, "--out-dir", str(tmp_path / "tok"),
                    "--takes", *TAKES, "--device", "cpu"])
    for t in TAKES:
        ids = torch.load(tmp_path / "tok" / f"{t}.pth.tar")
        poses = np.load(os.path.join(POSES, f"{t}.npy"))
        assert ids.dtype == torch.int64 and ids.shape == (len(poses), 34)
        assert int(ids.min()) >= 0 and int(ids.max()) < 2048
        # ExpertAF's loader: slice a 120-frame window, sample 10 frames, 34 ids each -> 10*34 pose tokens
        win = ids[100:220]
        idx = np.round(np.linspace(0, len(win) - 1, 10)).astype(int)
        assert win[idx].shape == (10, 34)
        m = summ["takes"][t]
        assert m["frames"] == len(poses) and m["frames_fully_visible"] == len(poses) and m["frames_all_invisible"] == 0
        assert 0 < m["pa_mm"] <= m["raw_mm"] < 2000          # alignment can only reduce the error
    assert summ["by_split"]["all"]["takes"] == 2
    # the ids rebuild to the same joints through decode_tokens, the path ExpertAF's scorer uses
    tok = ep.load_tokenizer(WEIGHTS, torch.device("cpu"))
    poses = torch.from_numpy(np.load(os.path.join(POSES, f"{TAKES[1]}.npy")).astype(np.float32)[:8])
    ids = torch.load(tmp_path / "tok" / f"{TAKES[1]}.pth.tar")[:8]
    joints, _, _ = tok(poses, train=False, return_embeds=False)
    assert torch.allclose(tok.decode_tokens(ids.view(-1), 8), joints, atol=1e-5)


def test_resume_and_tokenizer_mismatch(tmp_path):
    out = tmp_path / "tok"
    ep.main(["--pose-dir", POSES, "--weights", WEIGHTS, "--out-dir", str(out), "--takes", TAKES[1], "--device", "cpu"])
    first = torch.load(out / f"{TAKES[1]}.pth.tar")
    s2 = ep.main(["--pose-dir", POSES, "--weights", WEIGHTS, "--out-dir", str(out), "--takes", *TAKES, "--device", "cpu"])
    assert torch.equal(torch.load(out / f"{TAKES[1]}.pth.tar"), first) and set(s2["takes"]) == set(TAKES)
    other = tmp_path / "other.pth"
    other.write_bytes(open(WEIGHTS, "rb").read())
    with pytest.raises(SystemExit, match="new tokenizer needs a new out-dir"):
        ep.main(["--pose-dir", POSES, "--weights", str(other), "--out-dir", str(out), "--device", "cpu"])


def test_invisible_frames_still_get_tokens(tmp_path):
    d = tmp_path / "poses"
    d.mkdir()
    p = np.load(os.path.join(POSES, f"{TAKES[1]}.npy"))[:50].copy()
    p[10:20] = 0                                                       # a 10-frame gap: all joints invisible
    p[30, :5, 3] = 0                                                   # a frame with 5 invisible joints
    np.save(d / "t.npy", p)
    json.dump({}, open(d / "ignored.json", "w"))
    s = ep.main(["--pose-dir", str(d), "--weights", WEIGHTS, "--out-dir", str(tmp_path / "tok"), "--device", "cpu"])
    ids = torch.load(tmp_path / "tok" / "t.pth.tar")
    assert ids.shape == (50, 34) and s["takes"]["t"]["frames_all_invisible"] == 10
    assert s["takes"]["t"]["frames_fully_visible"] == 39
    assert torch.equal(ids[10], ids[15])                               # identical all-invisible frames -> identical ids
