"""train_pct_tokenizer.py: ExpertAF's tokenizer + PCT's loss/schedule, take selection, and the checkpoint format.

Uses REAL pose files: two takes exported from the EE4D-Motion reconstructions by the InterMimic repo's
scripts/export_ee4d_coco17.py (kept in ~/Downloads/expertaf_test_poses), the real InterMimic test clip list,
and ExpertAF's real llava/train/train.py config.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_train_pct_tokenizer.py -q
"""
import ast
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import train_pct_tokenizer as tp  # noqa: E402

POSES = os.path.expanduser("~/Downloads/expertaf_test_poses")
TEST_CLIPS = os.path.expanduser("~/InterMimic/docs/test_clips.csv")
TAKES = ["iiith_soccer_004_2", "sfu_basketball013_11"]
NO_RECON = "uniandes_bouldering_034_67"          # one of the 14 the real export could not reconstruct


def test_config_is_expertafs_everywhere():
    """The config read from train.py equals the literal parse_pose_generation.py uses to load the weights."""
    cfg = tp.expertaf_pose_config()
    tree = ast.parse(open(os.path.join(HERE, "parse_pose_generation.py")).read())
    lit = next(ast.literal_eval(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "pose_model_config")
    assert cfg == lit
    assert cfg["codebook"] == {"token_num": 34, "token_dim": 512, "token_class_num": 2048, "ema_decay": 0.9}


def test_joint_loss_is_pcts():
    gt = torch.zeros(1, 2, 4)
    gt[0, 0, 3] = 1.0                                   # joint 0 visible, joint 1 not
    pred = torch.zeros(1, 2, 3)
    pred[0, 0] = torch.tensor([0.01, 0.2, 0.0])         # quadratic and linear branches
    pred[0, 1] = torch.tensor([5.0, 5.0, 5.0])          # invisible: no loss, but still in the average
    q, l = 0.5 * 0.01 ** 2 / 0.05, 0.2 - 0.5 * 0.05
    assert torch.isclose(tp.joint_s1_loss(pred, gt), torch.tensor(((q + l + 0) / 3 + 0) / 2))


def test_schedule_matches_pct_config():
    assert tp.lr_at(0, 0, 150) == pytest.approx(tp.LR * tp.WARMUP_RATIO)
    assert tp.lr_at(500, 0, 150) == pytest.approx(tp.LR)
    assert tp.lr_at(10 ** 6, 150, 150) == pytest.approx(tp.LR * tp.MIN_LR_RATIO)


def _pose_dir(tmp_path, with_summary=True):
    d = tmp_path / "poses"
    d.mkdir()
    for t in TAKES:
        os.symlink(os.path.join(POSES, f"{t}.npy"), d / f"{t}.npy")
    if with_summary:
        json.dump({"takes": {}, "no_reconstruction": [NO_RECON]}, open(d / "export_summary.json", "w"))
    return d


def test_take_selection_drops_only_what_it_may(tmp_path):
    import csv
    first = next(csv.DictReader(open(TEST_CLIPS)))["take_name"]
    d = _pose_dir(tmp_path)
    rows = [{"video1": TAKES[0], "video2": TAKES[1]}, {"video1": NO_RECON, "video2": first}]
    json.dump(rows, open(tmp_path / "train.json", "w"))
    keep, dropped = tp.select_takes(tmp_path / "train.json", TEST_CLIPS, d)
    assert keep == TAKES
    assert dropped == {NO_RECON: "no EE4D-Motion reconstruction", first: "test clip list"}
    rows.append({"video1": "cmu_soccer99_9", "video2": TAKES[0]})          # no file, not excused
    json.dump(rows, open(tmp_path / "train.json", "w"))
    with pytest.raises(SystemExit, match="no pose file"):
        tp.select_takes(tmp_path / "train.json", TEST_CLIPS, d)


def test_smoke_run_writes_a_checkpoint_expertaf_loads(tmp_path):
    import csv
    first = next(csv.DictReader(open(TEST_CLIPS)))["take_name"]
    d = _pose_dir(tmp_path)
    json.dump([{"video1": TAKES[0], "video2": TAKES[1]}, {"video1": first, "video2": TAKES[0]}],
              open(tmp_path / "train.json", "w"))
    out = tmp_path / "out"
    loss, path = tp.main(["--pose-dir", str(d), "--train-json", str(tmp_path / "train.json"),
                          "--test-clips", TEST_CLIPS, "--out-dir", str(out), "--epochs", "2",
                          "--max-iters", "3", "--device", "cpu"])
    assert torch.isfinite(torch.tensor(loss)) and os.path.basename(path).startswith("best_avg_joint_loss_epoch_")
    split = json.load(open(out / "split.json"))
    assert len(split["train"]) == 1 and len(split["val"]) == 1 and split["dropped"] == {first: "test clip list"}
    # exactly how ExpertAF loads --pose_weights (parse_pose_generation.py): strip the prefix, strict=True
    w = torch.load(path, map_location="cpu")
    sd = {k[len(tp.PREFIX):] if k.startswith(tp.PREFIX) else k: v for k, v in w["state_dict"].items()}
    tok = tp.build_tokenizer()
    tok.load_state_dict(sd, strict=True)
    # a resumed run continues, a different take set is refused
    assert tp.main(["--pose-dir", str(d), "--train-json", str(tmp_path / "train.json"), "--test-clips", TEST_CLIPS,
                    "--out-dir", str(out), "--epochs", "3", "--max-iters", "2", "--device", "cpu"])
    with pytest.raises(SystemExit, match="different run"):
        tp.main(["--pose-dir", str(d), "--train-json", str(tmp_path / "train.json"), "--test-clips", TEST_CLIPS,
                 "--out-dir", str(out), "--seed", "1", "--device", "cpu"])
