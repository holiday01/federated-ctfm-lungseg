#!/usr/bin/env python3
"""Revision-1 inference-only analyses from existing checkpoints (no training).

For every requested run and every site it is evaluated on, on the seed's TEST split:

  A. Baseline pass on the cached ROI (reference bounding box + 80 mm). The recomputed Dice is
     checked against the value stored in the run JSON, then
       * component analysis of the raw prediction: components that overlap the reference (TP)
         versus components that do not (FP); FP count, FP volume, distance of each FP component
         from the reference;
       * simple filters: largest connected component (LCC) and minimum component size
         0.01 / 0.1 / 1 cm3; for each, Dice, tumor recall, the share of correctly segmented
         tumor voxels the filter removed, and whether the tumor was removed entirely;
       * calibration error in regions that do not depend on the prediction: the reference
         dilated by 10 voxels ("band") and the whole ROI ("roi"), next to the original
         region-conditional ECE (band OR prediction).

  B. With --perturb: less precise ROIs, all cut from inside the cached ROI so that no new
     preprocessing is involved: centered margins of 60 / 40 / 20 mm; a 40 mm margin shifted by
     20 or 40 mm in a random direction; and a "rough box" with six independent per-side margins
     drawn from [20, 80] mm. The random draws are keyed by case id, so every model sees the
     same boxes. For each box the model is run on the sub-volume; the Dice of the BASELINE
     prediction restricted to the same box is stored beside it, which separates the effect of
     the changed input from the mere exclusion of distant false positives from the scored region.

Output: results/fl_r1/roi/<tag>.json (per-case records). Usage:
  eval_roi_r1.py [--perturb] <tag> [<tag> ...]      or      --tags-file <file with one tag per line>
"""
from __future__ import annotations
import argparse, json, sys, time, zlib
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "fl_study"))
import train_fl as T  # noqa: E402

RES = ROOT / "results" / "fl"
CK = ROOT / "data" / "fl_ckpt"
OUT = ROOT / "results" / "fl_r1" / "roi"
MIN_SIZES = (10, 100, 1000)          # voxels at 1 mm3 = 0.01 / 0.1 / 1 cm3
BAND = 10


def dice_of(p, g):
    d = float(p.sum() + g.sum())
    return 2.0 * float((p & g).sum()) / d if d > 0 else 0.0


def ece_region(prob, gt, region, n_bins=10):
    p = prob[region]; y = gt[region].astype(np.float32)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def baseline_record(prob, gt):
    pred = prob >= 0.5
    gsum = float(gt.sum())
    rec = {"dice": dice_of(pred, gt), "gt_cm3": gsum / 1000.0, "pred_cm3": float(pred.sum()) / 1000.0,
           "recall": float((pred & gt).sum()) / gsum if gsum else float("nan")}
    band = ndimage.binary_dilation(gt, iterations=BAND)
    rec["ece_cond"] = ece_region(prob, gt, band | pred)          # the manuscript's original definition
    rec["ece_band"] = ece_region(prob, gt, band)                 # identical region for every model
    rec["ece_roi"] = ece_region(prob, gt, np.ones_like(gt, bool))
    lab, n = ndimage.label(pred)
    rec["n_comp"] = int(n)
    if n == 0:
        rec.update({"n_fp_comp": 0, "n_tp_comp": 0, "fp_cm3": 0.0, "fp_frac_of_pred": float("nan"), "largest_fp_cm3": 0.0,
                    "fp_dist_median_mm": float("nan"), "fp_vol_frac_ge20mm": float("nan"), "filters": {}})
        return rec, pred
    sizes = np.bincount(lab.ravel(), minlength=n + 1).astype(np.int64); sizes[0] = 0
    ov = np.bincount(lab[gt], minlength=n + 1); ov[0] = 0
    ids = np.arange(n + 1)
    fp_ids = ids[(ov == 0) & (ids > 0)]
    tp_ids = ids[ov > 0]
    fp_vox = float(sizes[fp_ids].sum())
    rec.update({"n_fp_comp": int(len(fp_ids)), "n_tp_comp": int(len(tp_ids)), "fp_cm3": fp_vox / 1000.0,
                "fp_frac_of_pred": fp_vox / float(pred.sum()), "largest_fp_cm3": float(sizes[fp_ids].max()) / 1000.0 if len(fp_ids) else 0.0})
    if len(fp_ids):
        dist = ndimage.distance_transform_edt(~gt)
        dmin = np.asarray(ndimage.minimum(dist, lab, index=fp_ids), dtype=np.float64)
        rec["fp_dist_median_mm"] = float(np.median(dmin))
        rec["fp_vol_frac_ge20mm"] = float(sizes[fp_ids][dmin >= 20].sum()) / fp_vox
        del dist
    else:
        rec["fp_dist_median_mm"] = float("nan"); rec["fp_vol_frac_ge20mm"] = float("nan")
    gt_c = T.drop_specks(gt)
    tp_before = float((pred & gt).sum())
    filt = {}
    variants = {"lcc": lab == int(np.argmax(sizes))}
    for ms in MIN_SIZES:
        variants[f"min{ms}"] = (sizes >= ms)[lab]
    for name, v in variants.items():
        tp_after = float((v & gt).sum())
        f = {"dice": dice_of(v, gt), "recall": tp_after / gsum if gsum else float("nan"),
             "tp_removed_frac": (tp_before - tp_after) / gsum if gsum else float("nan"),
             "tumor_lost": bool(tp_before > 0 and tp_after == 0),
             "n_comp": int(ndimage.label(v)[1]), "pred_cm3": float(v.sum()) / 1000.0}
        if name in ("lcc", "min100", "min1000"):
            f["hd95"] = T.hd95_mm(v, gt_c)
        filt[name] = f
    rec["hd95"] = T.hd95_mm(pred, gt_c)
    rec["filters"] = filt
    return rec, pred


def boxes_for(cid, gt):
    """Perturbed ROIs inside the cached volume; identical for every model (seeded by case id)."""
    rng = np.random.RandomState(zlib.crc32(cid.encode()) & 0xFFFFFFFF)
    shape = np.array(gt.shape)
    nz = np.argwhere(gt); lo = nz.min(0); hi = nz.max(0) + 1

    def box(mlo, mhi, shift=(0, 0, 0)):
        a = np.clip(lo - np.asarray(mlo) + np.asarray(shift), 0, shape)
        b = np.clip(hi + np.asarray(mhi) + np.asarray(shift), 0, shape)
        return np.minimum(a, lo).astype(int), np.maximum(b, hi).astype(int)     # the tumor always stays inside

    out = {}
    for m in (60, 40, 20):
        out[f"m{m}"] = box([m] * 3, [m] * 3)
    for s in (20, 40):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        out[f"m40s{s}"] = box([40] * 3, [40] * 3, np.round(s * u).astype(int))
    out["rough"] = box(rng.randint(20, 81, size=3), rng.randint(20, 81, size=3))
    return out, lo, hi


def load_model(tag, device):
    model, _ = T.build_model("scratch", 0, device)
    ckd = torch.load(CK / f"{tag}.pt", map_location=device)
    fedbn = isinstance(ckd, dict) and "client_bn" in ckd
    return model, ckd, fedbn


def process(tag, device, perturb):
    op = OUT / f"{tag}.json"
    if op.exists():
        print(f"{tag}: exists, skipped", flush=True); return
    r = json.load(open(RES / f"{tag}.json"))
    model, ckd, fedbn = load_model(tag, device)
    if not fedbn:
        model.load_state_dict(ckd)
    sites = {k: T.Site(k, r["seed"]) for k in T.SITE_ORDER}
    out = {"tag": tag, "seed": r["seed"], "init": r["init"], "mode": r["mode"], "variant": r["variant"], "train_sites": r["train_sites"],
           "perturb": bool(perturb), "min_sizes_vox": list(MIN_SIZES), "band_vox": BAND, "sites": {}}
    t0 = time.time(); diffs = []
    for site in T.SITE_ORDER:
        stored = r["results"].get(site, {})
        ref = stored.get("test") or stored.get("full")
        if ref is None:
            continue                                   # e.g. a local run evaluated without held-out sites
        heldout = "test" not in stored
        if r["mode"] == "local" and heldout:
            continue                                   # single-site transfer is not part of these analyses
        if fedbn:
            model.load_state_dict(T.merged_state(ckd["global"], ckd["client_bn"].get(site)))
        ref_dice = dict(zip(ref["ids"], ref["dice"]))
        recs = []
        for row in sites[site].split["test"]:
            cid = row["id"]
            img, msk = sites[site].arrays(row)
            gt = np.asarray(msk) > 0
            if gt.sum() == 0:
                continue
            img = np.asarray(img)
            prob = T.predict_prob(model, img, device)
            rec, pred = baseline_record(prob, gt)
            rec["id"] = cid
            if cid in ref_dice:
                diffs.append(abs(rec["dice"] - ref_dice[cid]))
            if perturb:
                bx, lo, hi = boxes_for(cid, gt)
                rec["perturb"] = {}
                for name, (a, b) in bx.items():
                    sl = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
                    g = gt[sl]
                    p = T.predict_prob(model, img[sl], device) >= 0.5
                    lab, n = ndimage.label(p)
                    if n > 1:
                        sz = np.bincount(lab.ravel()); sz[0] = 0; pl = lab == int(np.argmax(sz))
                    else:
                        pl = p
                    rec["perturb"][name] = {"dice": dice_of(p, g), "dice_lcc": dice_of(pl, g), "n_comp": int(n),
                                            "dice_base_inbox": dice_of(pred[sl], g),
                                            "margin_lo": [int(x) for x in (lo - a)], "margin_hi": [int(x) for x in (b - hi)]}
            recs.append(rec)
        out["sites"][site] = {"heldout": heldout, "cases": recs}
        print(f"  {tag} {site}{' (held out)' if heldout else ''}: n={len(recs)} dice={np.mean([c['dice'] for c in recs]):.4f} "
              f"fp_comp_median={np.median([c['n_fp_comp'] for c in recs]):.0f} ece_band={np.nanmean([c['ece_band'] for c in recs]):.4f}"
              + (f" rough={np.mean([c['perturb']['rough']['dice'] for c in recs]):.4f}" if perturb else ""), flush=True)
    diffs = np.array(diffs) if diffs else np.zeros(1)
    out["verify"] = {"n": int(len(diffs)), "max_abs_dice_diff": float(diffs.max()), "mean_abs_dice_diff": float(diffs.mean())}
    # same tolerance as eval_postproc.py (fp16 sliding-window inference is not bit-reproducible)
    assert diffs.max() < 5e-3 and diffs.mean() < 5e-4, f"{tag}: recomputed Dice differs from stored ({out['verify']})"
    out["time_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w"), indent=1)
    print(f"== {tag}: {out['verify']} {out['time_s']/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--tags-file")
    ap.add_argument("--perturb", action="store_true")
    a = ap.parse_args()
    tags = list(a.tags)
    if a.tags_file:
        tags += [t.strip() for t in open(a.tags_file) if t.strip() and not t.startswith("#")]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    for tag in tags:
        process(tag, device, a.perturb)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
