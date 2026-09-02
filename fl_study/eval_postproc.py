#!/usr/bin/env python3
"""Post-hoc re-evaluation of finished runs from their checkpoints.

Two purposes:
  1. Independent verification: recompute every stored per-case Dice from the checkpoint with the
     shared evaluation code and assert it matches the value in the run JSON (tolerance 1e-3, the
     residual coming from cuDNN autotuning non-determinism).
  2. Back-fill: for runs written before metrics_version 2 (HD95 = max of directed 95th percentiles,
     LCC metrics), recompute hd95, dice_lcc and hd95_lcc from the checkpoint and overwrite those
     keys only; a 'postproc' provenance block records what was done and metrics_version is set.

Usage: python fl_study/eval_postproc.py <tag> [<tag> ...]   (or --missing to back-fill every run lacking dice_lcc)
"""
import glob, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "fl_study"))
import train_fl as T  # noqa: E402

RES = ROOT / "results" / "fl"


def process(tag, device):
    jp = RES / f"{tag}.json"
    r = json.load(open(jp))
    ck = ROOT / "data" / "fl_ckpt" / f"{tag}.pt"
    model, _ = T.build_model("scratch", 0, device)
    ckd = torch.load(ck, map_location=device)
    fedbn = isinstance(ckd, dict) and "client_bn" in ckd      # FedBN: own BN at training sites, averaged BN elsewhere
    model.load_state_dict(ckd["global"] if fedbn else ckd)
    sites = {k: T.Site(k, r["seed"]) for k in T.SITE_ORDER}
    max_abs = 0.0; n = 0; diffs = []
    for site, d in r["results"].items():
        if fedbn:
            model.load_state_dict(T.merged_state(ckd["global"], ckd["client_bn"].get(site)))
        for split, res in d.items():
            rows = {row["id"]: row for row in sites[site].split[split]}
            new = {"dice_lcc": [], "hd95_lcc": [], "hd95": []}
            for i, cid in enumerate(res["ids"]):
                img, msk = sites[site].arrays(rows[cid])
                m = T.case_metrics(T.predict_prob(model, img, device), msk, full=True)
                diffs.append(abs(m["dice"] - res["dice"][i])); max_abs = max(max_abs, diffs[-1]); n += 1
                new["dice_lcc"].append(m["dice_lcc"]); new["hd95_lcc"].append(m["hd95_lcc"]); new["hd95"].append(m["hd95"])
            res.update(new)
            print(f"  {tag} {site}/{split}: n={len(res['ids'])} dice_lcc={np.mean(new['dice_lcc']):.4f} hd95_lcc={np.nanmedian(new['hd95_lcc']):.1f} (raw dice {np.mean(res['dice']):.4f}, hd95 {np.nanmedian(res['hd95']):.1f})", flush=True)
    for site, sp in r.get("personalized", {}).items():
        pass  # personalised results keep their original metric set
    diffs = np.array(diffs)
    print(f"  {tag}: |dDice| mean {diffs.mean():.2e}, p99 {np.percentile(diffs, 99):.2e}, max {max_abs:.2e} over {n} cases", flush=True)
    # fp16 sliding-window inference under cuDNN autotuning is not bit-reproducible; a few voxels near
    # p = 0.5 can flip. Tolerate per-case |dDice| up to 5e-3 and a mean below 5e-4; anything larger is a bug.
    assert max_abs < 5e-3 and diffs.mean() < 5e-4, f"{tag}: recomputed Dice differs from stored (max {max_abs:.4f}, mean {diffs.mean():.5f})"
    for site, d in r["summary"].items():
        for split, e in d.items():
            res = r["results"][site][split]
            e["dice_lcc_mean"] = float(np.mean(res["dice_lcc"])); e["hd95_lcc_median"] = float(np.nanmedian(res["hd95_lcc"]))
            e["hd95_median"] = float(np.nanmedian(res["hd95"])); e["empty_pred_frac"] = float(np.mean(np.isnan(res["hd95"])))
    r["postproc"] = {"recomputed": ["hd95", "dice_lcc", "hd95_lcc"], "verified_cases": n, "max_abs_dice_diff": max_abs,
                     "mean_abs_dice_diff": float(diffs.mean()), "p99_abs_dice_diff": float(np.percentile(diffs, 99)), "timestamp": time.strftime("%Y%m%d_%H%M%S")}
    r["metrics_version"] = T.METRICS_VERSION
    json.dump(r, open(jp, "w"), indent=1)
    print(f"== {tag}: verified {n} cases, max |dDice| = {max_abs:.2e}; back-filled LCC metrics", flush=True)


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    tags = sys.argv[1:]
    if tags == ["--missing"]:
        tags = []
        for f in sorted(glob.glob(str(RES / "*.json"))):
            try:
                r = json.load(open(f))
            except Exception:  # noqa: BLE001
                continue
            if "results" in r and not r["tag"].startswith(("probe_", "smoke_")) and r.get("metrics_version", 1) < T.METRICS_VERSION:
                tags.append(r["tag"])
        print("runs below metrics_version", T.METRICS_VERSION, ":", tags)
    for tag in tags:
        process(tag, device)


if __name__ == "__main__":
    main()
