#!/usr/bin/env python3
"""Build the unified per-site cache used by fl_study/train_fl.py.

Every site ends up as data/fl_cache/<site>/<id>_img.npy (int16, 1 mm iso, HU in [-1000,400]),
<id>_msk.npy (uint8) and manifest.json listing id, patient, shape and tumour volume. Uncompressed
.npy files are memory-mapped by the trainer, so three sites (~14 GB) stay out of process memory.

Sites:
  TW  in-house (data/processed_seg_1mm, QC-clean allowlist), patient = case id with _B<n> stripped
  NL  NSCLC-Radiomics (data/external_nsclc_processed_1mm), patient = case
  US  NSCLC Radiogenomics (data/external_radiogenomics_processed_1mm), patient = case
"""
import argparse, glob, json, os, re
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path("/home/holiday/lung_ct")
CACHE = ROOT / "data" / "fl_cache"
SITES = {
    "TW": dict(dir=ROOT / "data" / "processed_seg_1mm", glob="seg_*.npz", clean=True),
    "NL": dict(dir=ROOT / "data" / "external_nsclc_processed_1mm", glob="ext_*.npz", clean=False),
    "US": dict(dir=ROOT / "data" / "external_radiogenomics_processed_1mm", glob="rg_*.npz", clean=False),
}


def patient_of(site, case):
    if site == "TW":
        return re.sub(r"_B\d+$", "", case)
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="TW,NL,US")
    a = ap.parse_args()
    summary = {}
    for site in a.sites.split(","):
        cfg = SITES[site]
        files = sorted(glob.glob(str(cfg["dir"] / cfg["glob"])))
        if cfg["clean"]:
            cl = json.load(open(ROOT / "data" / "qc_clean_manifest.json"))
            good = set(cl["sets"]["seg_1mm"]["good"])
            before = len(files)
            files = [f for f in files if os.path.basename(f) in good]
            print(f"{site}: QC allowlist kept {len(files)}/{before}")
        if not files:
            print(f"{site}: no files yet, skipped"); continue
        out = CACHE / site; out.mkdir(parents=True, exist_ok=True)
        rows = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            img, msk, case = z["image"].astype(np.int16), (z["mask"] > 0).astype(np.uint8), str(z["case"])
            cid = os.path.splitext(os.path.basename(f))[0]
            ip, mp = out / f"{cid}_img.npy", out / f"{cid}_msk.npy"
            if not (ip.exists() and mp.exists()):
                np.save(ip, img); np.save(mp, msk)
            lab, ncomp = ndimage.label(msk)
            sizes = np.bincount(lab.ravel())[1:] if ncomp else np.array([0])
            rows.append(dict(id=cid, case=case, patient=patient_of(site, case), shape=list(img.shape),
                             tumour_cm3=float(msk.sum()) / 1000.0, n_components=int(ncomp),
                             largest_component_frac=float(sizes.max() / max(1, msk.sum())) if ncomp else 0.0))
        json.dump(rows, open(out / "manifest.json", "w"), indent=1)
        vols = np.array([r["tumour_cm3"] for r in rows]); shapes = np.array([r["shape"] for r in rows])
        summary[site] = dict(n_cases=len(rows), n_patients=len({r["patient"] for r in rows}),
                             tumour_cm3_median=float(np.median(vols)), tumour_cm3_iqr=[float(np.percentile(vols, 25)), float(np.percentile(vols, 75))],
                             tumour_cm3_min=float(vols.min()), tumour_cm3_max=float(vols.max()),
                             frac_lt1cm3=float((vols < 1).mean()), frac_gt50cm3=float((vols > 50).mean()),
                             crop_shape_median=[float(x) for x in np.median(shapes, 0)],
                             multi_component_frac=float(np.mean([r["n_components"] > 1 for r in rows])),
                             cache_gb=float(sum(np.prod(r["shape"]) * 3 for r in rows) / 1e9))
        print(site, json.dumps(summary[site]))
    (ROOT / "results" / "fl").mkdir(parents=True, exist_ok=True)
    sp = ROOT / "results" / "fl" / "site_summary.json"
    old = json.load(open(sp)) if sp.exists() else {}
    old.update(summary)
    json.dump(old, open(sp, "w"), indent=2)
    print("wrote", sp)


if __name__ == "__main__":
    main()
