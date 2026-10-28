#!/usr/bin/env python3
"""Retrain ExpertAF's 3D PCT pose tokenizer (its weights, best_avg_joint_loss_epoch_150.pth, were never released).

What comes from where -- nothing here is our own choice except the take list and the validation split:
  * MODEL: ExpertAF's llava/model/multimodal_encoder/pct_tokenizer.py, unchanged (PCT [Geng et al., CVPR 2023]
    adapted by ExpertAF to 3D joints: start_embed takes xyz, recover_embed returns xyz), built exactly as
    ExpertAF builds it (builder.build_pose_model: stage "tokenizer", 17 joints, guide_ratio 0) with ExpertAF's
    pose_model_config (llava/train/train.py ModelArguments default; checked by the test).
  * LOSS: PCT's official Tokenizer_loss (github Gengzigang/PCT models/pct_loss.py): smooth-L1 on the rebuilt
    joints (beta 0.05) masked by visibility and averaged over ALL joints, x joint_loss_w 1.0, plus the codebook
    commitment term e_latent_loss x e_loss_w 15.0 (configs/pct_base_tokenizer.py). ExpertAF's own get_loss
    calls an undefined self.loss, so the loss cannot come from ExpertAF's file.
  * OPTIMISER / SCHEDULE: PCT's tokenizer config -- AdamW lr 1e-2, betas (0.9, 0.999), weight decay 0.15, no
    grad clip; linear warm-up over 500 iterations from 0.001 x lr, then cosine annealing by epoch to 1e-5 x lr
    (mmcv CosineAnnealing, by_epoch); 64 poses per GPU on 8 GPUs = 512 per step (here one GPU, batch 512).
  * EPOCHS / SELECTION: 150 epochs, keep the epoch with the lowest validation joint loss -- ExpertAF's checkpoint
    is named best_avg_joint_loss_epoch_150.pth (PCT's own default is 50).
  * DATA: one pose = one frame (17, 4) = world-frame xyz in metres + visibility, fed RAW, as ExpertAF's loader
    feeds its .npy (llava/train/train.py: np.load -> slice -> tokenizer, no normalisation). Poses = our
    EE4D-Motion COCO-17 export (scripts/export_ee4d_coco17.py in the InterMimic repo); ExpertAF's own pose files
    are not public. Frames with no visible joint are skipped.
  * TAKES (Jess 2026-10-28): every take in ExpertAF's train.json (video1 and video2), MINUS every take in our
    test clip list, MINUS the takes the export found no reconstruction for (export_summary.json: 14 bouldering
    takes of session uniandes_bouldering_034). Any OTHER take without a pose file is an error, never skipped.
    Validation = a seeded 5% of those takes (whole takes, so no frame of a validation take is trained on).

Output (--out-dir): split.json (train / val takes, dropped takes and why), log.csv (one row per epoch),
last.pth (for resuming this run only), and best_avg_joint_loss_epoch_<E>.pth in the format ExpertAF loads:
{'state_dict': {'keypoint_head.tokenizer.<param>': ...}} (pretrain_v15.sh --pose_weights). Refuses an out-dir
that holds another run's split; resumes from its own last.pth.

    python train_pct_tokenizer.py --pose-dir <ee4d_coco17 dir> --train-json <train.json> \
        --test-clips <InterMimic>/docs/test_clips.csv --out-dir <out>
"""
import argparse
import ast
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# The tokenizer module needs only torch + timm; import it by file so the LLaVA package (transformers,
# deepspeed) is not pulled in.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "pct_tokenizer", os.path.join(HERE, "llava/model/multimodal_encoder/pct_tokenizer.py"))
pct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pct)

# PCT configs/pct_base_tokenizer.py
LR, BETAS, WEIGHT_DECAY = 1e-2, (0.9, 0.999), 0.15
WARMUP_ITERS, WARMUP_RATIO, MIN_LR_RATIO = 500, 0.001, 1e-5
JOINT_LOSS_W, E_LOSS_W, BETA = 1.0, 15.0, 0.05
BATCH = 512                       # 64 per GPU x 8 GPUs
PREFIX = "keypoint_head.tokenizer."


def expertaf_pose_config():
    """ExpertAF's pose_model_config, read from the default of ModelArguments.pose_model_config in
    llava/train/train.py (parsed, not imported: importing train.py needs the full LLaVA stack)."""
    tree = ast.parse(open(os.path.join(HERE, "llava/train/train.py")).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "pose_model_config":
            lam = next(k.value for k in node.value.keywords if k.arg == "default_factory")
            return ast.literal_eval(lam.body)
    raise SystemExit("ERROR: pose_model_config default not found in llava/train/train.py")


def build_tokenizer():
    """As llava/model/multimodal_encoder/builder.py build_pose_model('pct_tokenizer', ...)."""
    return pct.PCT_Tokenizer("tokenizer", expertaf_pose_config(), num_joints=17, guide_ratio=0, guide_channels=0)


def joint_s1_loss(pred, gt):
    """PCT JointS1Loss: smooth-L1 per coordinate x visibility, mean over coords, joints (all, visible or not), batch."""
    d = gt.shape[2] - 1
    vis = gt[..., d:]
    l1 = torch.abs(pred[..., :d] - gt[..., :d])
    loss = torch.where(l1 < BETA, 0.5 * l1 ** 2 / BETA, l1 - 0.5 * BETA) * vis
    return loss.mean(dim=2).mean(dim=1).mean(dim=0)


def lr_at(it, epoch, epochs):
    """mmcv CosineAnnealing(by_epoch) with linear warm-up over the first WARMUP_ITERS iterations."""
    min_lr = LR * MIN_LR_RATIO
    lr = min_lr + (LR - min_lr) * (1 + math.cos(math.pi * epoch / epochs)) / 2
    if it < WARMUP_ITERS:
        lr *= 1 - (1 - WARMUP_RATIO) * (1 - it / WARMUP_ITERS)
    return lr


def select_takes(train_json, test_clips, pose_dir):
    """(takes, dropped {take: reason}). Missing pose files are allowed ONLY for the takes the export
    recorded as having no reconstruction; anything else is an error."""
    takes = sorted({t for e in json.load(open(train_json)) for t in (e["video1"], e["video2"])})
    test = {r["take_name"] for r in csv.DictReader(open(test_clips))}
    summ_path = os.path.join(pose_dir, "export_summary.json")
    if not os.path.isfile(summ_path):
        raise SystemExit(f"ERROR: {summ_path} missing -- the pose dir is not a finished export")
    no_recon = set(json.load(open(summ_path))["no_reconstruction"])
    keep, dropped, missing = [], {}, []
    for t in takes:
        if t in test:
            dropped[t] = "test clip list"
        elif t in no_recon:
            dropped[t] = "no EE4D-Motion reconstruction"
        elif not os.path.isfile(os.path.join(pose_dir, f"{t}.npy")):
            missing.append(t)
        else:
            keep.append(t)
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} train.json takes have no pose file and are not listed as "
                         f"unreconstructable: {missing[:10]}")
    return keep, dropped


def split_takes(takes, val_frac, seed):
    rng = random.Random(seed)
    shuffled = takes[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(takes) * val_frac))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def load_frames(pose_dir, takes):
    """(N, 17, 4) float32 of every frame with at least one visible joint."""
    out = []
    for t in takes:
        p = np.load(os.path.join(pose_dir, f"{t}.npy"))
        if p.ndim != 3 or p.shape[1:] != (17, 4):
            raise SystemExit(f"ERROR: {t}.npy has shape {p.shape}, expected (T, 17, 4)")
        out.append(p[p[..., 3].sum(axis=1) > 0])
    return torch.from_numpy(np.concatenate(out).astype(np.float32))


def init_single_process_group(device):
    """The tokenizer's EMA codebook update calls dist.all_reduce (PCT trains on 8 GPUs); one process here."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", rank=0, world_size=1)


def run_epoch(model, frames, opt, device, it, epoch, epochs, train, max_iters=None):
    """One pass; returns (mean joint loss, mean e-latent loss, mean per-joint error in mm on visible joints, it)."""
    model.train(train)
    idx = torch.randperm(len(frames)) if train else torch.arange(len(frames))
    sums, n = np.zeros(3), 0
    for b, s in enumerate(range(0, len(frames), BATCH)):
        if max_iters is not None and b >= max_iters:
            break
        x = frames[idx[s:s + BATCH]].to(device)
        if train:
            for g in opt.param_groups:
                g["lr"] = lr_at(it, epoch, epochs)
        with torch.set_grad_enabled(train):
            rec, _, e_latent = model(x, train=train, return_embeds=False)
            jl = joint_s1_loss(rec, x) * JOINT_LOSS_W
            el = e_latent * E_LOSS_W
            if train:
                opt.zero_grad(set_to_none=True)
                (jl + el).backward()
                opt.step()
                it += 1
        vis = x[..., 3]
        mm = ((rec - x[..., :3]).norm(dim=-1) * vis).sum() / vis.sum().clamp(min=1) * 1000
        sums += [jl.item(), el.item(), mm.item()]
        n += 1
    return (*(sums / max(n, 1)), it)


def save_expertaf(model, path, epoch, val):
    sd = {PREFIX + k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "epoch": epoch, "val_joint_loss": val}, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--test-clips", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-iters", type=int, help="smoke test: batches per epoch")
    ap.add_argument("--takes", nargs="*", help="smoke test: use only these takes (still filtered)")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    takes, dropped = select_takes(a.train_json, a.test_clips, a.pose_dir)
    if a.takes:
        takes = [t for t in takes if t in set(a.takes)]
    tr, va = split_takes(takes, a.val_frac, a.seed)
    split = {"train": tr, "val": va, "dropped": dropped, "pose_dir": os.path.abspath(a.pose_dir), "seed": a.seed}
    os.makedirs(a.out_dir, exist_ok=True)
    sp = os.path.join(a.out_dir, "split.json")
    if os.path.exists(sp) and json.load(open(sp)) != split:
        raise SystemExit(f"ERROR: {sp} belongs to a different run (other takes / seed) -- use a new --out-dir")
    json.dump(split, open(sp, "w"), indent=1)
    reasons = {r: sum(v == r for v in dropped.values()) for r in set(dropped.values())}
    print(f"[pct] takes: {len(tr)} train, {len(va)} val; dropped {reasons}")

    train_f, val_f = load_frames(a.pose_dir, tr), load_frames(a.pose_dir, va)
    print(f"[pct] frames: {len(train_f)} train, {len(val_f)} val; {math.ceil(len(train_f) / BATCH)} steps/epoch")

    init_single_process_group(device)
    model = build_tokenizer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    start, it, best = 0, 0, (float("inf"), None)
    last = os.path.join(a.out_dir, "last.pth")
    if os.path.isfile(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, it, best = ck["epoch"], ck["it"], tuple(ck["best"])
        print(f"[pct] RESUMING at epoch {start} (iteration {it}); best so far {best}")

    log = os.path.join(a.out_dir, "log.csv")
    new_log = not os.path.isfile(log)
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["epoch", "lr", "train_joint_loss", "train_e_loss", "train_mm", "val_joint_loss", "val_e_loss",
                        "val_mm", "seconds"])
        for ep in range(start, a.epochs):
            t0 = time.time()
            tj, te, tmm, it = run_epoch(model, train_f, opt, device, it, ep, a.epochs, True, a.max_iters)
            vj, ve, vmm, _ = run_epoch(model, val_f, None, device, it, ep, a.epochs, False, a.max_iters)
            w.writerow([ep + 1, f"{lr_at(it, ep, a.epochs):.3e}", f"{tj:.5f}", f"{te:.5f}", f"{tmm:.1f}",
                        f"{vj:.5f}", f"{ve:.5f}", f"{vmm:.1f}", f"{time.time() - t0:.0f}"])
            fh.flush()
            print(f"[pct] epoch {ep + 1}/{a.epochs}: train joint {tj:.4f} ({tmm:.1f} mm)  val joint {vj:.4f} "
                  f"({vmm:.1f} mm)  e {ve:.4f}", flush=True)
            if vj < best[0]:
                if best[1] and os.path.isfile(best[1]):
                    os.remove(best[1])           # keep only the best, as ExpertAF's single checkpoint
                path = os.path.join(a.out_dir, f"best_avg_joint_loss_epoch_{ep + 1}.pth")
                save_expertaf(model, path, ep + 1, vj)
                best = (vj, path)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "it": it,
                        "best": list(best)}, last)
    print(f"[pct] done: best {best[1]} (val joint loss {best[0]:.5f})")
    return best


if __name__ == "__main__":
    main()
