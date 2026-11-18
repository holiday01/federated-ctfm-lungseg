#!/usr/bin/env python3
"""Test-only evaluation of EXISTING checkpoints on an external cohort (first journal revision, analysis J).

The cohort (fl_cache_ext/<COHORT>, built by preprocess_ext.py) took no part in training, model
selection or any design decision. Inference is identical to train_fl.evaluate: sliding window over
the cached ROI (reference bounding box + 80 mm), threshold 0.5, no post-processing.

Per case: Dice, tumor recall, precision, reference and predicted volume, number of predicted
components, Dice of the largest predicted component, detection (Dice > 0.1); with --hd95 also the
HD95 of the raw and the largest-component prediction (same definition as train_fl.hd95_mm).

Output: results/fl_r1/ext/<COHORT>/<tag>.json. Usage:
  eval_ext.py <COHORT> [--tags-file F | tag ...] [--hd95] [--max-cases N --no-write  (smoke test)]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "fl_study"))
import train_fl as T  # noqa: E402

RES = ROOT / "results" / "fl"
CK = ROOT / "data" / "fl_ckpt"
EXT = Path("/data2/holiday/lung_ext/fl_cache_ext")
OUT = ROOT / "results" / "fl_r1" / "ext"


def metrics(prob, msk, hd95):
    gt = np.asarray(msk) > 0; pred = prob >= 0.5
    inter = float((pred & gt).sum()); ps, gs = float(pred.sum()), float(gt.sum())
    out = {"dice": 2 * inter / (ps + gs) if ps + gs > 0 else 0.0, "recall": inter / gs, "precision": inter / ps if ps > 0 else float("nan"),
           "gt_cm3": gs / 1000.0, "pred_cm3": ps / 1000.0}
    out["detected"] = bool(out["dice"] > 0.1)
    lab, n = ndimage.label(pred)
    out["n_pred_components"] = int(n)
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        pred_l = lab == int(np.argmax(sizes))
    else:
        pred_l = pred
    il = float((pred_l & gt).sum()); dl = float(pred_l.sum()) + gs
    out["dice_lcc"] = 2 * il / dl if dl > 0 else 0.0
    if hd95:
        gt_c = T.drop_specks(gt)
        out["hd95"] = T.hd95_mm(pred, gt_c); out["hd95_lcc"] = T.hd95_mm(pred_l, gt_c)
    return out


def process(cohort, tag, rows, device, hd95, write):
    op = OUT / cohort / f"{tag}.json"
    if write and op.exists():
        print(f"{tag}: exists, skipped", flush=True); return
    r = json.load(open(RES / f"{tag}.json"))
    ckd = torch.load(CK / f"{tag}.pt", map_location=device)
    assert not (isinstance(ckd, dict) and "client_bn" in ckd), "FedBN checkpoints have no normalization state for an unseen site"
    model, _ = T.build_model("scratch", 0, device)
    model.load_state_dict(ckd)
    res = {"ids": []}
    t0 = time.time()
    for row in rows:
        img = np.load(EXT / cohort / f"{row['id']}_img.npy", mmap_mode="r")
        msk = np.load(EXT / cohort / f"{row['id']}_msk.npy", mmap_mode="r")
        m = metrics(T.predict_prob(model, img, device), msk, hd95)
        res["ids"].append(row["id"])
        for k, v in m.items():
            res.setdefault(k, []).append(v)
    dt = time.time() - t0
    print(f"  {cohort} {tag}: n={len(res['ids'])} dice={np.mean(res['dice']):.4f} lcc={np.mean(res['dice_lcc']):.4f} recall={np.mean(res['recall']):.4f} "
          f"det={np.mean(res['detected']):.3f} comps_median={np.median(res['n_pred_components']):.0f} {dt/len(res['ids']):.1f} s/case", flush=True)
    if not write:
        return
    (OUT / cohort).mkdir(parents=True, exist_ok=True)
    json.dump({"tag": tag, "cohort": cohort, "seed": r["seed"], "init": r["init"], "mode": r["mode"], "variant": r["variant"], "train_sites": r["train_sites"],
               "iters_per_site": r["config"].get("iters_per_site"), "rounds": r["config"].get("rounds"), "hd95": bool(hd95), "n": len(res["ids"]),
               "results": res, "time_s": dt}, open(op, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort")
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--tags-file")
    ap.add_argument("--hd95", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    tags = list(a.tags)
    if a.tags_file:
        tags += [t.strip() for t in open(a.tags_file) if t.strip() and not t.startswith("#")]
    rows = json.load(open(EXT / a.cohort / "manifest.json"))
    if a.max_cases:
        rows = rows[:a.max_cases]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    for tag in tags:
        process(a.cohort, tag, rows, device, a.hd95, not a.no_write)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
