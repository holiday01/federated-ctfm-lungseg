#!/usr/bin/env python3
"""Structure of the reference masks per site (Reviewer 3, major 3): connected components of each
mask, with and without components below ten voxels, and the share of the mask volume that lies
outside its largest component. Reads data/fl_cache only. Writes results/fl_r1/mask_structure.json."""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path("/home/holiday/lung_ct"); CACHE = ROOT / "data" / "fl_cache"
out = {}
for site in ("TW", "NL", "US"):
    rows = json.load(open(CACHE / site / "manifest.json"))
    rec = []
    for r in rows:
        m = np.load(CACHE / site / f"{r['id']}_msk.npy") > 0
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        sz = np.sort(np.bincount(lab.ravel())[1:])[::-1]
        big = sz[sz >= 10]
        rec.append({"n": int(n), "n_ge10": int(len(big)), "n_ge100": int((sz >= 100).sum()),
                    "outside_largest": float(1 - sz[0] / sz.sum()),
                    "second_cm3": float(sz[1]) / 1000 if n > 1 else 0.0})
    a = lambda k: np.array([x[k] for x in rec])
    out[site] = {"n_masks": len(rec), "n_comp_median": float(np.median(a("n"))), "n_comp_ge10_median": float(np.median(a("n_ge10"))),
                 "frac_masks_multi": float(np.mean(a("n") > 1)), "frac_masks_multi_ge10": float(np.mean(a("n_ge10") > 1)),
                 "frac_masks_multi_ge100": float(np.mean(a("n_ge100") > 1)),
                 "outside_largest_median": float(np.median(a("outside_largest"))), "outside_largest_mean": float(np.mean(a("outside_largest"))),
                 "frac_masks_outside_gt5pct": float(np.mean(a("outside_largest") > 0.05)),
                 "second_component_cm3_p90": float(np.percentile(a("second_cm3"), 90))}
    print(site, json.dumps(out[site]))
(ROOT / "results" / "fl_r1").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "results" / "fl_r1" / "mask_structure.json", "w"), indent=1)
