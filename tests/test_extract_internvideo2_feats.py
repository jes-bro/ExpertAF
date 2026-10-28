"""extract_internvideo2_feats.py on REAL inputs: ExpertAF's real train/test JSONs, the real Ego-Exo4D
takes.json, and real Ego-Exo4D 448p frame-aligned videos (~/exp_3_input_videos: aria01_214-1.mp4 448x448 +
cam01..cam04 796x448, 30 fps, 1634 frames) laid out under a take's path as the cluster has them.

The InternVideo2 model itself is not run here (gated 1B checkpoint + flash-attn, cluster only): the file-writing
path is exercised with a STAND-IN encoder (a deterministic function of the pixels with the model's output
contract, (16, 512) unit-norm). The real encoder is exercised by the cluster smoke run (--takes <one take>).

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_extract_internvideo2_feats.py -q
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import extract_internvideo2_feats as ex  # noqa: E402

JSON_DIR = os.path.expanduser("~/Downloads/expertaf_json")
TAKES_JSON = os.path.expanduser("~/egoexo4d/takes.json")
VIDEOS = os.path.expanduser("~/exp_3_input_videos")
TAKE = "sfu_basketball015_2"            # a real take whose best_exo is cam01; the local videos stand under its path


def stand_in_encoder(x):
    """(16, 8, 3, 224, 224) -> (16, 512) unit-norm, a fixed function of the pixels (NOT the model)."""
    assert tuple(x.shape) == (16, 8, 3, 224, 224) and x.dtype == torch.float32
    v = x.mean(dim=(1, 3, 4))                                  # (16, 3)
    f = torch.cat([v] * 171, dim=1)[:, :512]
    return torch.nn.functional.normalize(f, dim=-1)


def layout(tmp_path):
    d = tmp_path / "egoexo" / "takes" / TAKE / "frame_aligned_videos" / "downscaled" / "448"
    d.mkdir(parents=True)
    for f in ("aria01_214-1.mp4", "cam01.mp4"):
        os.symlink(os.path.join(VIDEOS, f), d / f)
    return tmp_path / "egoexo"


def jsons(tmp_path, starts, test_starts=()):
    d = tmp_path / "json"
    d.mkdir()
    rows = [{"video1": TAKE, "video1_range": f"[{s}, {s + 120}]", "video2": TAKE, "video2_range": f"[{s}, {s + 120}]"}
            for s in starts]
    json.dump(rows, open(d / "train.json", "w"))
    json.dump([{"video1": TAKE, "video1_range": f"[{s}, {s + 120}]", "video2": TAKE, "video2_range": f"[{s}, {s + 120}]"}
               for s in test_starts], open(d / "test.json", "w"))
    return d


def test_frame_indices_are_32fps_points_inside_the_4s_window():
    idx = ex.frame_indices(500)
    assert len(idx) == 128 and idx[0] == 500 and idx[-1] == 619          # 4 s = 120 source frames
    assert all(b >= a for a, b in zip(idx, idx[1:]))
    spans = [idx[8 * j + 7] - idx[8 * j] for j in range(16)]
    assert all(6 <= s <= 7 for s in spans)                              # 8 points at 32 fps = 7.5 frames at 30 fps


def test_real_windows_and_best_exo():
    w = ex.windows_from_jsons(JSON_DIR)
    assert len(w) == 557 and sum(len(v) for v in w.values()) == 11566
    exo = ex.best_exo_map(TAKES_JSON, w)
    assert len(exo) == 557 and all(c.startswith("cam") for c in exo.values())


def test_frames_match_an_independent_read_and_demo_normalisation(tmp_path):
    root = layout(tmp_path)
    path = ex.video_path(root, TAKE, "aria")
    idx = ex.frame_indices(300)
    frames = ex.read_frames(path, idx)
    assert set(frames) == set(idx) and frames[idx[0]].shape == (224, 224, 3) and frames[idx[0]].dtype == np.uint8
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx[40])
    ok, bgr = cap.read()
    assert ok
    ref = cv2.resize(bgr[:, :, ::-1], (224, 224))
    assert np.array_equal(frames[idx[40]], ref)
    x = ex.chunk_tensor(frames, idx)
    assert tuple(x.shape) == (16, 8, 3, 224, 224) and x.dtype == torch.float32
    # chunk 5, point 0 = idx[40]; undo (p/255 - mean)/std
    back = (x[5, 0].permute(1, 2, 0).numpy() * ex.V_STD + ex.V_MEAN) * 255.0
    assert np.abs(back - ref).max() < 1e-2


def test_main_writes_expertaf_feature_files_resumes_and_shards(tmp_path):
    root = layout(tmp_path)
    jd = jsons(tmp_path, [0, 500, 1500], test_starts=[1000])
    out = tmp_path / "feats"
    args = ["--expertaf-json-dir", str(jd), "--takes-json", TAKES_JSON, "--egoexo-root", str(root), "--out-dir", str(out)]
    s = ex.main(args, encoder=stand_in_encoder)
    assert s["windows_written"] == 4 and s["windows_skipped"] == 0
    for st in (0, 500, 1000, 1500):
        for v in ("aria", "exo"):
            f = torch.load(out / f"{TAKE}_{st}_{v}.pth.tar")
            assert tuple(f.shape) == (16, 512) and f.dtype == torch.float32
            assert torch.allclose(f.norm(dim=-1), torch.ones(16), atol=1e-4)
    a0 = torch.load(out / f"{TAKE}_500_aria.pth.tar")
    assert not torch.equal(a0, torch.load(out / f"{TAKE}_500_exo.pth.tar"))        # two different videos
    s2 = ex.main(args, encoder=stand_in_encoder)                                    # resume: nothing rewritten
    assert s2["windows_written"] == 0 and s2["windows_skipped"] == 4
    assert torch.equal(a0, torch.load(out / f"{TAKE}_500_aria.pth.tar"))
    # extra windows CSV adds a window; sharding with one take: shard 1 of 2 gets nothing
    (tmp_path / "extra.csv").write_text(f"take,start\n{TAKE},700\n")
    s3 = ex.main(args + ["--windows", str(tmp_path / "extra.csv")], encoder=stand_in_encoder)
    assert s3["windows_written"] == 1 and os.path.isfile(out / f"{TAKE}_700_exo.pth.tar")
    s4 = ex.main(args + ["--shard", "1", "--num-shards", "2"], encoder=stand_in_encoder)
    assert s4["takes"] == {} and s4["windows_written"] == 0


def test_window_past_the_end_of_the_video_is_an_error(tmp_path):
    root = layout(tmp_path)
    jd = jsons(tmp_path, [1520])                                                   # 1520 + 119 > 1634 frames
    with pytest.raises(SystemExit, match="runs past the end"):
        ex.main(["--expertaf-json-dir", str(jd), "--takes-json", TAKES_JSON, "--egoexo-root", str(root),
                 "--out-dir", str(tmp_path / "f")], encoder=stand_in_encoder)
