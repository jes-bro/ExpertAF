#!/usr/bin/env python3
"""InternVideo2 ego/exo features for every ExpertAF window, in the files ExpertAF's loader reads.

ExpertAF reads, per sample, egoexo_path/<take>_<start>_aria.pth.tar and <take>_<start>_exo.pth.tar
(llava/train/train.py), one tensor each, stacked over the batch and fed to a Linear(512 -> hidden) projector
(mm_hidden_size = 512, llava_arch.py). Those files were never released ("taken from" a folder inside
InternVideo2's demo/ on the authors' cluster, things_to_provide.md), so this reproduces them from the paper:

  Appendix C: "The demonstration clips are 4 seconds long and we sample frames at 32 fps"; the encoder
  "takes in 8 frames and thus, generates 4 features per second (totalling 16 features)"; ego and exo
  features are concatenated into 32 tokens. Sec 3.3 footnote: the exo view is Ego-Exo4D's best_exo.

So one window = 4 s from the 30 fps frame `start` of ExpertAF's range: 128 time points at 32 fps
(k/32 s, k = 0..127), each mapped to the nearest source frame start + round(k * 30/32); 16 chunks of 8
consecutive points; each chunk -> ONE 512-d vector = InternVideo2-Stage2's own get_vid_feat (the demo's
class: pooled vision embedding -> vision_proj -> L2 norm; frames resized to 224x224, BGR->RGB, ImageNet
mean/std exactly as demo/utils.py frames2tensor does). Output per view: float32 (16, 512).

Model: the public InternVideo2-stage2_1b-224p-f4.pt (trained with 4 frames per clip); fed 8 frames per the
paper by InternVideo2's own positional-embedding interpolation (config origin_num_frames = 4 -> num_frames 8,
setup_internvideo2). This 4 -> 8 step is a judgment call (the paper says 8 frames, the released checkpoint is
f4): stated in the paper. Everything else is the demo code's defaults (fp32, no flash-attention kernels --
the backbone still IMPORTS flash_attn, so the env needs it installed).

Videos: <egoexo-root>/takes/<take>/frame_aligned_videos/downscaled/448/aria01_214-1.mp4 (ego) and
.../<best_exo>.mp4 (exo), 30 fps, frame-aligned with the poses. A take's video is decoded once per view for
all its windows. Windows: every (video1, range start) and (video2, range start) of train.json + test.json,
plus optional extra (take,start) rows (--windows CSV, for our own test clips later). Sharded by take
(--shard i --num-shards n) for an array job; existing files are skipped (resumable); every written tensor
is checked to be (16, 512) and unit-norm. Writes <out-dir>/feat_summary_<shard>.json.

    python extract_internvideo2_feats.py --expertaf-json-dir <dir> --takes-json <takes.json> \
        --egoexo-root /vision/group/egoexo4d --internvideo-root <InternVideo>/InternVideo2/multi_modality \
        --weights <InternVideo2-stage2_1b-224p-f4.pt> --out-dir <egoexo_feats> [--shard 0 --num-shards 8]
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

WINDOW_FRAMES, SRC_FPS, FPS, N_POINTS, CHUNK = 120, 30, 32, 128, 8
N_CHUNKS, FEAT_DIM, SIZE = N_POINTS // CHUNK, 512, 224
V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)      # demo/utils.py
V_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
EGO_FILE = "aria01_214-1.mp4"
VIEWS = ("aria", "exo")


def windows_from_jsons(json_dir, extra_csv=None):
    """{take: sorted list of window start frames} from ExpertAF's train/test pairs (+ extra rows)."""
    win = {}
    for split in ("train", "test"):
        p = os.path.join(json_dir, f"{split}.json")
        if not os.path.isfile(p):
            raise SystemExit(f"ERROR: missing {p}")
        for e in json.load(open(p)):
            for v in ("video1", "video2"):
                win.setdefault(e[v], set()).add(int(eval(e[f"{v}_range"])[0]))
    if extra_csv:
        for r in csv.DictReader(open(extra_csv)):
            win.setdefault(r["take"], set()).add(int(r["start"]))
    return {t: sorted(s) for t, s in sorted(win.items())}


def best_exo_map(takes_json, takes):
    """{take: best_exo cam}; a take without one is an error (ExpertAF uses exactly that view)."""
    info = {x["take_name"]: x for x in json.load(open(takes_json))}
    out, bad = {}, []
    for t in takes:
        cam = (info.get(t) or {}).get("best_exo")
        (out.__setitem__(t, cam) if cam else bad.append(t))
    if bad:
        raise SystemExit(f"ERROR: {len(bad)} takes have no best_exo in takes.json: {bad[:10]}")
    return out


def video_path(root, take, view, cam=None):
    name = EGO_FILE if view == "aria" else f"{cam}.mp4"
    return os.path.join(root, "takes", take, "frame_aligned_videos", "downscaled", "448", name)


def frame_indices(start):
    """The 128 source-frame indices of a window: 32 fps time points mapped to the nearest 30 fps frame."""
    return [start + int(round(k * SRC_FPS / FPS)) for k in range(N_POINTS)]


def read_frames(path, needed):
    """{frame index: (224, 224, 3) uint8 RGB} for the needed indices, decoding the video once, in order."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {path}")
    need, out, last = set(needed), {}, max(needed)
    i = 0
    while i <= last:
        ok = cap.grab()
        if not ok:
            break
        if i in need:
            ok, frame = cap.retrieve()
            if not ok:
                break
            out[i] = cv2.resize(frame[:, :, ::-1], (SIZE, SIZE))      # BGR -> RGB, then 224x224 (demo order)
        i += 1
    cap.release()
    missing = sorted(need - set(out))
    if missing:
        raise SystemExit(f"ERROR: {path}: {len(missing)} needed frames beyond the video's {i} frames "
                         f"(first {missing[:3]}) -- a window runs past the end of the take")
    return out


def chunk_tensor(frames, idx):
    """(16, 8, 3, 224, 224) float32, normalised as demo/utils.py frames2tensor: (x/255 - mean) / std."""
    x = np.stack([(frames[i] / 255.0 - V_MEAN) / V_STD for i in idx]).astype(np.float32)   # (128, H, W, 3)
    x = x.reshape(N_CHUNKS, CHUNK, SIZE, SIZE, 3).transpose(0, 1, 4, 2, 3)                # B, T, C, H, W
    return torch.from_numpy(np.ascontiguousarray(x))


def load_model(internvideo_root, weights, device):
    """InternVideo2_Stage2 from the repo's demo code, config demo/internvideo2_stage2_config.py with the
    checkpoint path and 8 frames (origin_num_frames 4 -> pos-embed interpolation in setup_internvideo2)."""
    root = os.path.abspath(internvideo_root)
    for p in (root, os.path.join(root, "demo")):
        if p not in sys.path:
            sys.path.insert(0, p)
    cfg_mod = _load_by_path(os.path.join(root, "demo_config.py"), "demo_config")
    utils = _load_by_path(os.path.join(root, "demo", "utils.py"), "iv2_demo_utils")
    config = cfg_mod.eval_dict_leaf(cfg_mod.Config.from_file(os.path.join(root, "demo", "internvideo2_stage2_config.py")))
    if not os.path.isfile(weights):
        raise SystemExit(f"ERROR: checkpoint not found: {weights}")
    config.pretrained_path = weights
    config.device = str(device)
    config.num_frames = config.num_frames_test = CHUNK
    config.model.vision_encoder.num_frames = CHUNK
    if config.get("origin_num_frames", None) != 4:
        raise SystemExit(f"ERROR: expected origin_num_frames 4 in the demo config, got {config.get('origin_num_frames')}")
    print(f"[iv2] config: frames {config.model.vision_encoder.num_frames} (checkpoint {config.origin_num_frames}, "
          f"pos-embed interpolated), size {config.get('size_t')}, embed {config.model.embed_dim}, "
          f"half {config.get('use_half_precision')}, flash {config.model.vision_encoder.get('use_flash_attn')}", flush=True)
    model, _ = utils.setup_internvideo2(config)
    return model


def _load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def encode_with(model, x, device):
    """One window: (16, 8, 3, 224, 224) -> (16, 512) via the model's own get_vid_feat."""
    f = model.get_vid_feat(x.to(device))
    return f.float().cpu()


def check_feat(f, what):
    if tuple(f.shape) != (N_CHUNKS, FEAT_DIM):
        raise SystemExit(f"ERROR: {what}: feature shape {tuple(f.shape)}, expected {(N_CHUNKS, FEAT_DIM)}")
    n = f.norm(dim=-1)
    if not torch.allclose(n, torch.ones_like(n), atol=1e-3):
        raise SystemExit(f"ERROR: {what}: features are not unit-norm (get_vid_feat normalises)")


def run(windows, exo, args, encoder):
    """encoder(x: (16,8,3,224,224) float) -> (16,512). Returns the summary dict."""
    takes = sorted(windows)
    mine = takes[args.shard::args.num_shards]
    os.makedirs(args.out_dir, exist_ok=True)
    summ = {"shard": args.shard, "num_shards": args.num_shards, "takes": {}, "windows_written": 0, "windows_skipped": 0}
    t0 = time.time()
    for ti, take in enumerate(mine):
        starts = windows[take]
        todo = [s for s in starts if not all(os.path.isfile(os.path.join(args.out_dir, f"{take}_{s}_{v}.pth.tar")) for v in VIEWS)]
        summ["windows_skipped"] += len(starts) - len(todo)
        if not todo:
            continue
        for view in VIEWS:
            path = video_path(args.egoexo_root, take, view, exo[take])
            if not os.path.isfile(path):
                raise SystemExit(f"ERROR: missing video {path}")
            idx = {s: frame_indices(s) for s in todo}
            frames = read_frames(path, sorted({i for v in idx.values() for i in v}))
            for s in todo:
                out = os.path.join(args.out_dir, f"{take}_{s}_{view}.pth.tar")
                if os.path.isfile(out):
                    continue
                f = encoder(chunk_tensor(frames, idx[s]))
                check_feat(f, out)
                torch.save(f.contiguous(), out + ".tmp")
                os.replace(out + ".tmp", out)
            del frames
        summ["windows_written"] += len(todo)
        summ["takes"][take] = {"windows": len(starts), "exo": exo[take]}
        print(f"  [{ti + 1}/{len(mine)}] {take}: {len(todo)} windows x 2 views ({time.time() - t0:.0f} s)", flush=True)
    json.dump(summ, open(os.path.join(args.out_dir, f"feat_summary_{args.shard}.json"), "w"), indent=1)
    print(f"[iv2] shard {args.shard}/{args.num_shards}: {len(mine)} takes, {summ['windows_written']} windows written, "
          f"{summ['windows_skipped']} already there, {time.time() - t0:.0f} s", flush=True)
    return summ


def main(argv=None, encoder=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expertaf-json-dir", required=True)
    ap.add_argument("--windows", help="extra CSV with columns take,start (30 fps frame)")
    ap.add_argument("--takes-json", required=True)
    ap.add_argument("--egoexo-root", required=True)
    ap.add_argument("--internvideo-root", help="InternVideo/InternVideo2/multi_modality (needed unless an encoder is injected)")
    ap.add_argument("--weights", help="InternVideo2-stage2_1b-224p-f4.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--takes", nargs="*", help="only these takes (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if not 0 <= a.shard < a.num_shards:
        ap.error("--shard must be in [0, --num-shards)")

    windows = windows_from_jsons(a.expertaf_json_dir, a.windows)
    if a.takes:
        missing = sorted(set(a.takes) - set(windows))
        if missing:
            raise SystemExit(f"ERROR: no windows for {missing}")
        windows = {t: windows[t] for t in a.takes}
    exo = best_exo_map(a.takes_json, windows)
    n_win = sum(len(v) for v in windows.values())
    print(f"[iv2] {len(windows)} takes, {n_win} windows ({n_win * 2} feature files); this shard: "
          f"{len(sorted(windows)[a.shard::a.num_shards])} takes", flush=True)
    if encoder is None:
        if not (a.internvideo_root and a.weights):
            ap.error("--internvideo-root and --weights are required")
        model = load_model(a.internvideo_root, a.weights, torch.device(a.device))
        encoder = lambda x: encode_with(model, x, torch.device(a.device))  # noqa: E731
    return run(windows, exo, a, encoder)


if __name__ == "__main__":
    main()
