#!/usr/bin/env python3
"""Whole-cohort check of the slice order of the labels of an external cohort (first revision, analysis J).

In ESLS26 every label has the size and spacing of its CT series, but its header puts the first slice at the
cranial end (z direction -1), while SimpleITK reads the DICOM series caudal to cranial. Two readings are possible:
  header  the header is right: the label array is the CT array reversed along z;
  asis    the header is wrong: the label array is in the order of the CT array.
Per case this script records
  - the geometry (sizes equal, z direction of the label, where its origin lies),
  - the InstanceNumber of the most cranial and of the most caudal DICOM slice (the data descriptor says the
    labels follow the InstanceNumber order),
  - for both readings the contrast of the mask against its surroundings: mean HU inside the mask minus mean HU
    in a 4-voxel shell around it. A nodule is denser than the lung around it, so the right reading has the
    larger contrast; a misplaced mask has a contrast near zero or negative.
Reads untrusted data: run with `python -I`. Writes one JSON; prints counts only.
Usage: python -I diag_ext_orient.py ESLS26 /path/out.json [--workers 8]
"""
import glob, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage

SRC = Path("/data2/holiday/lung_ext/anhui_scidb29318")


def contrast(ia, m):
    if m.sum() == 0:
        return None
    nz = np.argwhere(m); lo = np.maximum(nz.min(0) - 6, 0); hi = np.minimum(nz.max(0) + 7, m.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    mm, ii = m[sl], ia[sl]
    shell = ndimage.binary_dilation(mm, iterations=4) & ~mm
    if shell.sum() == 0:
        return None
    return float(ii[mm].mean() - ii[shell].mean()), float(ii[mm].mean()), float(ii[shell].mean())


def one(args):
    cohort, pid = args
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
        pdir = SRC / cohort / cohort / pid
        series = sorted({os.path.dirname(f) for f in glob.glob(str(pdir / "*" / "*" / "*.dcm"))})
        labels = sorted(glob.glob(str(pdir / "*" / "*.nii.gz")) + glob.glob(str(pdir / "*" / "*.nii")))
        if len(labels) != 1 or len(series) != 1:
            return {"case": pid, "error": f"{len(labels)} labels, {len(series)} series"}
        rdr = sitk.ImageSeriesReader(); files = rdr.GetGDCMSeriesFileNames(series[0]); rdr.SetFileNames(files)
        img = rdr.Execute(); lab = sitk.ReadImage(labels[0])
        ia = sitk.GetArrayFromImage(img).astype(np.float32); la = sitk.GetArrayFromImage(lab) > 0
        zdir = float(np.array(lab.GetDirection()).reshape(3, 3)[2, 2])
        z0 = img.GetOrigin()[2]; z1 = z0 + (img.GetSize()[2] - 1) * img.GetSpacing()[2]
        inst = lambda f: int(getattr(pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["InstanceNumber"]), "InstanceNumber", -1))
        r = {"case": pid, "same_size": bool(lab.GetSize() == img.GetSize()), "same_spacing": bool(np.allclose(lab.GetSpacing(), img.GetSpacing(), atol=1e-3)),
             "same_xy_origin": bool(np.allclose(lab.GetOrigin()[:2], img.GetOrigin()[:2], atol=0.5)), "label_zdir": zdir,
             "label_origin_at": "cranial" if abs(lab.GetOrigin()[2] - z1) < 0.5 else "caudal" if abs(lab.GetOrigin()[2] - z0) < 0.5 else "other",
             "instance_caudal": inst(files[0]), "instance_cranial": inst(files[-1]), "n_slices": int(img.GetSize()[2]), "label_voxels": int(la.sum())}
        if r["same_size"]:
            for k, m in (("header", la[::-1] if zdir < 0 else la), ("asis", la)):
                c = contrast(ia, m)
                r[k] = None if c is None else {"contrast": c[0], "inside": c[1], "shell": c[2]}
        return r
    except Exception as e:  # noqa: BLE001
        return {"case": pid, "error": f"{type(e).__name__}: {e}"}


def main():
    cohort, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    root = SRC / cohort / cohort
    pids = sorted(p for p in os.listdir(root) if (root / p).is_dir())
    with Pool(workers) as pool:
        rows = list(pool.imap_unordered(one, [(cohort, p) for p in pids], chunksize=4))
    rows.sort(key=lambda r: r["case"])
    json.dump(rows, open(out, "w"), indent=1)
    ok = [r for r in rows if "error" not in r]
    cnt = lambda f: int(sum(bool(f(r)) for r in ok))
    print("cases", len(rows), "| read", len(ok), "| errors", len(rows) - len(ok))
    print("same size", cnt(lambda r: r["same_size"]), "| same spacing", cnt(lambda r: r["same_spacing"]), "| same x/y origin", cnt(lambda r: r["same_xy_origin"]))
    print("label z direction -1:", cnt(lambda r: r["label_zdir"] < 0), "| label origin cranial / caudal / other:",
          cnt(lambda r: r["label_origin_at"] == "cranial"), cnt(lambda r: r["label_origin_at"] == "caudal"), cnt(lambda r: r["label_origin_at"] == "other"))
    print("InstanceNumber 1 at the cranial end:", cnt(lambda r: r["instance_cranial"] < r["instance_caudal"]), "| at the caudal end:", cnt(lambda r: r["instance_cranial"] > r["instance_caudal"]))
    both = [r for r in ok if r.get("header") and r.get("asis")]
    h = np.array([r["header"]["contrast"] for r in both]); a = np.array([r["asis"]["contrast"] for r in both])
    print("contrast available:", len(both))
    print("header reading: median contrast %.0f HU (IQR %.0f to %.0f), contrast > 50 HU in %d" % (np.median(h), *np.percentile(h, [25, 75]), int((h > 50).sum())))
    print("as-is reading:  median contrast %.0f HU (IQR %.0f to %.0f), contrast > 50 HU in %d" % (np.median(a), *np.percentile(a, [25, 75]), int((a > 50).sum())))
    print("header contrast larger than as-is:", int((h > a).sum()), "| smaller:", int((h < a).sum()), "| equal (identical placement):", int((h == a).sum()))
    weak = [r["case"] for r, x, y in zip(both, h, a) if x <= y]
    print("cases where the header reading is not the larger:", len(weak), weak[:12])
    ci = np.array([r["instance_cranial"] < r["instance_caudal"] for r in both])
    print("among InstanceNumber-1-cranial cases: header larger in %d of %d; among the others: %d of %d" % (int((h > a)[ci].sum()), int(ci.sum()), int((h > a)[~ci].sum()), int((~ci).sum())))


if __name__ == "__main__":
    main()
