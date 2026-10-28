#!/usr/bin/env python3
"""Pose tokens for every take, in the files ExpertAF's pose-generation mode reads, plus the tokenizer's
rebuild error measured the way ExpertAF scores poses.

ExpertAF's loader (llava/train/train.py, --pose_generation): pose_token_path/<take>.pth.tar is loaded with
torch.load, sliced by the window's 30 fps frame range, 10 frames sampled uniformly, and each frame's 34
ids become <CUSTOM_POSE_TOKEN_xxxx> text (ids must be < 2048). So each file here is a LongTensor (T, 34)
indexed by the take's 30 fps frame number, T = the pose file's length: frame t's tokens = the tokenizer's
codebook ids for frame t of <pose-dir>/<take>.npy (the EE4D-Motion COCO-17 export, (T, 17, 4) xyz + vis).

Tokenizer = ExpertAF's PCT_Tokenizer built and loaded exactly as parse_pose_generation.py does (prefix
'keypoint_head.tokenizer.' stripped, strict=True), from the retrained checkpoint (train_pct_tokenizer.py).
Frames are fed RAW, as ExpertAF feeds them. Joints with visibility 0 are replaced by the tokenizer's
learned invisible token (its own forward); a frame outside the reconstruction (all joints invisible) still
gets 34 ids, deterministic, so a window that touches a gap never crashes the loader -- the export summary
counts such frames per take.

REBUILD ERROR (the floor under ExpertAF's pose error): for every frame the tokenizer's decoder rebuilds the
17 joints from the 34 ids; the script reports, per take and per split,
  raw_mm  mean joint distance on visible joints, no alignment (room coordinates, what the tokenizer sees)
  pa_mm   mean joint distance after ExpertAF's OWN Procrustes alignment (reconstruction_error in
          parse_pose_generation.py, pulled from that file by its source so it is their code), on frames
          with all 17 joints visible -- the metric ExpertAF reports (135 mm in the paper for GENERATED poses)
Also checks on the first batch that decode_tokens(ids) == the forward's rebuilt joints (the two decode paths
ExpertAF uses agree). Resumable: existing .pth.tar are skipped, their metrics kept from the summary.

    python export_pose_tokens.py --pose-dir <ee4d_coco17> --weights <best_avg_joint_loss_epoch_N.pth> \
        --out-dir <pose_tokens> [--split <train run's split.json>]
"""
import argparse
import ast
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_pct_tokenizer as tp  # noqa: E402  (build_tokenizer, PREFIX, expertaf_pose_config)


def expertaf_procrustes():
    """compute_similarity_transform + reconstruction_error exactly as parse_pose_generation.py defines them
    (that file runs a whole evaluation at import, so the two functions are compiled from its source)."""
    src = open(os.path.join(HERE, "parse_pose_generation.py")).read()
    tree = ast.parse(src)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("compute_similarity_transform", "reconstruction_error")]
    if len(defs) != 2:
        raise SystemExit("ERROR: parse_pose_generation.py no longer defines both Procrustes functions")
    ns = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=defs, type_ignores=[]), "parse_pose_generation.py", "exec"), ns)
    return ns["reconstruction_error"]


def load_tokenizer(weights, device):
    """As parse_pose_generation.py loads --pose_weights."""
    tok = tp.build_tokenizer()
    w = torch.load(weights, map_location="cpu")
    sd = {k[len(tp.PREFIX):] if k.startswith(tp.PREFIX) else k: v for k, v in w["state_dict"].items()}
    tok.load_state_dict(sd, strict=True)
    return tok.to(device).eval()


@torch.no_grad()
def tokenize_take(tok, poses, device, batch=2048):
    """(T, 34) long ids and (T, 17, 3) rebuilt joints for a (T, 17, 4) pose array."""
    ids, rec = [], []
    for s in range(0, len(poses), batch):
        x = torch.from_numpy(poses[s:s + batch]).to(device)
        joints, idx, _ = tok(x, train=False, return_embeds=False)
        ids.append(idx.view(len(x), tok.token_num).cpu())
        rec.append(joints.float().cpu())
    return torch.cat(ids), torch.cat(rec)


def take_metrics(poses, rec, pa):
    """raw / PA errors in mm, frame counts. poses (T,17,4) numpy, rec (T,17,3) tensor."""
    x = torch.from_numpy(poses)
    vis = x[..., 3] > 0
    d = (rec - x[..., :3]).norm(dim=-1)
    n_vis = int(vis.sum())
    raw = float((d * vis).sum() / max(n_vis, 1) * 1000)
    full = vis.all(dim=1)
    pa_mm = float(pa(rec[full], x[full][..., :3]).mean() * 1000) if int(full.sum()) else float("nan")
    return {"frames": len(poses), "frames_fully_visible": int(full.sum()),
            "frames_all_invisible": int((~vis).all(dim=1).sum()), "raw_mm": round(raw, 2), "pa_mm": round(pa_mm, 2)}


def check_decode_paths(tok, poses):
    """decode_tokens (what parse_pose_generation.py calls on generated ids) must equal the forward's rebuild."""
    cpu = load_state_copy(tok)
    x = torch.from_numpy(poses[: min(64, len(poses))])
    joints, idx, _ = cpu(x, train=False, return_embeds=False)
    via_tokens = cpu.decode_tokens(idx.view(-1), len(x))
    if not torch.allclose(joints, via_tokens, atol=1e-5):
        raise SystemExit("ERROR: decode_tokens(ids) != forward rebuild -- the two decode paths disagree")


def load_state_copy(tok):
    c = tp.build_tokenizer()
    c.load_state_dict(tok.state_dict())
    return c.eval()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", help="split.json of the tokenizer run: per-split (train/val) summary")
    ap.add_argument("--takes", nargs="*", help="only these takes (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    takes = sorted(f[:-4] for f in os.listdir(a.pose_dir) if f.endswith(".npy"))
    if a.takes:
        missing = sorted(set(a.takes) - set(takes))
        if missing:
            raise SystemExit(f"ERROR: no pose file for {missing}")
        takes = sorted(a.takes)
    if not takes:
        raise SystemExit(f"ERROR: no .npy pose files in {a.pose_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    summ_path = os.path.join(a.out_dir, "token_export_summary.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {"weights": os.path.abspath(a.weights), "takes": {}}
    if summ["weights"] != os.path.abspath(a.weights):
        raise SystemExit(f"ERROR: {a.out_dir} holds tokens from {summ['weights']} -- a new tokenizer needs a new out-dir")

    tok = load_tokenizer(a.weights, torch.device(a.device))
    pa = expertaf_procrustes()
    print(f"[tokens] {len(takes)} takes; tokenizer {a.weights}; {tok.token_num} tokens x {tok.token_class_num} codes", flush=True)
    checked = False
    t0 = time.time()
    for i, t in enumerate(takes):
        out = os.path.join(a.out_dir, f"{t}.pth.tar")
        if os.path.exists(out) and t in summ["takes"]:
            continue
        poses = np.load(os.path.join(a.pose_dir, f"{t}.npy")).astype(np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (17, 4):
            raise SystemExit(f"ERROR: {t}.npy has shape {poses.shape}, expected (T, 17, 4)")
        if not checked:
            check_decode_paths(tok, poses)
            checked = True
        ids, rec = tokenize_take(tok, poses, torch.device(a.device))
        if int(ids.max()) >= tok.token_class_num or int(ids.min()) < 0:
            raise SystemExit(f"ERROR: {t}: token id out of range")
        torch.save(ids.long(), out)
        m = take_metrics(poses, rec, pa)
        summ["takes"][t] = m
        print(f"  [{i + 1}/{len(takes)}] {t}: T={m['frames']} raw {m['raw_mm']:.1f} mm  PA {m['pa_mm']:.1f} mm"
              + (f"  ({m['frames_all_invisible']} frames outside the reconstruction)" if m["frames_all_invisible"] else ""),
              flush=True)
        if (i + 1) % 25 == 0:
            json.dump(summ, open(summ_path, "w"), indent=1)

    groups = {"all": takes}
    if a.split:
        sp = json.load(open(a.split))
        groups["train"], groups["val"] = [t for t in sp["train"] if t in summ["takes"]], [t for t in sp["val"] if t in summ["takes"]]
    summ["by_split"] = {}
    for g, ts in groups.items():
        ms = [summ["takes"][t] for t in ts if t in summ["takes"]]
        if not ms:
            continue
        w = np.array([m["frames_fully_visible"] for m in ms], dtype=float)
        pa_vals = np.array([m["pa_mm"] for m in ms], dtype=float)
        ok = w > 0
        summ["by_split"][g] = {"takes": len(ms),
                               "raw_mm_frame_weighted": round(float(np.average([m["raw_mm"] for m in ms], weights=[m["frames"] for m in ms])), 2),
                               "pa_mm_frame_weighted": round(float(np.average(pa_vals[ok], weights=w[ok])), 2) if ok.any() else None}
        print(f"[tokens] {g}: {summ['by_split'][g]}", flush=True)
    json.dump(summ, open(summ_path, "w"), indent=1)
    print(f"[tokens] done in {time.time() - t0:.0f} s -> {a.out_dir}", flush=True)
    return summ


if __name__ == "__main__":
    main()
