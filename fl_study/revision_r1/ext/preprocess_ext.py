#!/usr/bin/env python3
"""External TEST-ONLY cohorts into the format of data/fl_cache (first journal revision, analysis J).

Source: ESLS26 / ASLS26, Anhui Chest Hospital, Science Data Bank doi:10.57760/sciencedb.29318 (version 4, CC0);
descriptor Zhou et al., Scientific Data 2026, doi:10.1038/s41597-026-07963-0. Each patient folder holds one DICOM
CT series and one NIfTI label on the same grid.

Per patient, exactly the steps of fl_study/preprocess_radiogenomics.py:
  read the CT series (SimpleITK) and the label; require the same voxel grid (identical size and spacing, voxel
  centres coinciding; a label stored with an axis in the opposite direction is reordered to the CT);
  reorient to LPS; resample the image (linear, air fill) and the mask (nearest) to 1 mm isotropic;
  clip HU to [-1000, 400]; crop to the reference bounding box + 80 mm.
Writes <out>/<COHORT>/<id>_img.npy (int16), <id>_msk.npy (uint8), rows/<id>.json, a QC montage per case and
manifest.json (id, patient, shape, tumor volume, component structure, acquisition tags, series UID).
The cohort is never used for training or model selection.

The downloaded files are untrusted data: run with `python -I`, paths are passed as arguments.
Usage: python -I preprocess_ext.py ASLS26 [--workers 8] [--limit N]
"""
import argparse, glob, json, os, sys, traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "scripts"))
import preprocess as P  # noqa: E402

EXT = Path("/data2/holiday/lung_ext")
SRC = EXT / "anhui_scidb29318"
OUT = EXT / "fl_cache_ext"
SPACING = (1.0, 1.0, 1.0)
HU_MIN, HU_MAX = -1000, 400
HU_AIR = getattr(P, "HU_AIR", -1024)
MARGIN = 80
TAGS = {"manufacturer": "Manufacturer", "model": "ManufacturerModelName", "kernel": "ConvolutionKernel", "kvp": "KVP",
        "slice_thickness": "SliceThickness", "contrast": "ContrastBolusAgent", "series_desc": "SeriesDescription",
        "patient_position": "PatientPosition", "series_uid": "SeriesInstanceUID", "study_uid": "StudyInstanceUID"}


def same_grid(lab, img, tol=0.5):
    """True when lab and img describe the same set of voxel centres: same size and spacing, and every corner voxel
    of the label lies on a corner voxel of the CT (the axes may run in the opposite direction)."""
    if lab.GetSize() != img.GetSize() or float(np.max(np.abs(np.array(lab.GetSpacing()) - np.array(img.GetSpacing())))) > 1e-3:
        return False
    n = [s - 1 for s in img.GetSize()]
    corners = [tuple(int(b) * int(k) for b, k in zip(bits, n)) for bits in ((i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1))]
    pc = np.array([img.TransformIndexToPhysicalPoint(c) for c in corners])
    pl = np.array([lab.TransformIndexToPhysicalPoint(c) for c in corners])
    return all(float(np.min(np.linalg.norm(pc - q, axis=1))) <= tol for q in pl)


def montage(ia, ma, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nz = np.argwhere(ma)
    zc = int(np.median(nz[:, 0])); yc = int(np.argmax(ma.sum(axis=(0, 2))))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    ax[0].imshow(ia[zc], cmap="gray", vmin=-1000, vmax=400); ax[0].set_title(f"{title} axial (no contour)"); ax[0].axis("off")
    ax[1].imshow(ia[zc], cmap="gray", vmin=-1000, vmax=400); ax[1].contour(ma[zc], levels=[0.5], colors="r", linewidths=0.7)
    ax[1].set_title("axial + reference"); ax[1].axis("off")
    ax[2].imshow(ia[:, yc, :], cmap="gray", vmin=-1000, vmax=400, origin="lower")
    ax[2].contour(ma[:, yc, :], levels=[0.5], colors="r", linewidths=0.7); ax[2].set_title("coronal + reference"); ax[2].axis("off")
    plt.savefig(path, dpi=70, bbox_inches="tight"); plt.close(fig)


def one(args):
    cohort, pid = args
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    out = OUT / cohort
    cid = f"{cohort.lower()}_{pid.replace('-', '_')}"
    rp = out / "rows" / f"{cid}.json"
    if rp.exists() and (out / f"{cid}_img.npy").exists() and (out / f"{cid}_msk.npy").exists():
        return json.load(open(rp))
    try:
        pdir = SRC / cohort / cohort / pid
        series = sorted({os.path.dirname(f) for f in glob.glob(str(pdir / "*" / "*" / "*.dcm"))})
        labels = sorted(glob.glob(str(pdir / "*" / "*.nii.gz")) + glob.glob(str(pdir / "*" / "*.nii")))
        if len(labels) != 1 or not series:
            raise RuntimeError(f"expected one label and at least one series, found {len(labels)} / {len(series)}")
        rdr = sitk.ImageSeriesReader()
        cands = [(sd, sid, rdr.GetGDCMSeriesFileNames(sd, sid)) for sd in series for sid in rdr.GetGDCMSeriesIDs(sd)]
        lab = sitk.ReadImage(labels[0])
        match = [c for c in cands if len(c[2]) == lab.GetSize()[2]]
        if len(match) != 1:
            raise RuntimeError(f"{len(cands)} series, {len(match)} with the slice count of the label ({lab.GetSize()[2]})")
        files = match[0][2]
        rdr.SetFileNames(files)
        img = rdr.Execute()
        hdr = pydicom.dcmread(files[0], stop_before_pixels=True)
        meta = {k: str(getattr(hdr, t, "")) for k, t in TAGS.items()}
        if lab.GetSize() != img.GetSize():
            raise RuntimeError(f"label size {lab.GetSize()} != CT size {img.GetSize()}")
        geo = max(float(np.max(np.abs(np.array(lab.GetOrigin()) - np.array(img.GetOrigin())))),
                  float(np.max(np.abs(np.array(lab.GetDirection()) - np.array(img.GetDirection())))) * 100.0,
                  float(np.max(np.abs(np.array(lab.GetSpacing()) - np.array(img.GetSpacing())))) * 100.0)
        reoriented = False
        if geo > 0.5:
            # ESLS26: the label holds the voxel grid of the CT with the slices in the opposite order, as its header
            # states (z direction -1, origin at the cranial end). Such a label is brought into the order of the CT
            # through physical space; nearest neighbour on coinciding voxel centres moves no value between voxels.
            if not same_grid(lab, img):
                raise RuntimeError(f"label grid differs from the CT grid (score {geo:.3f})")
            n_before = int((sitk.GetArrayFromImage(lab) > 0).sum())
            lab = sitk.Resample(lab, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, lab.GetPixelID())
            if int((sitk.GetArrayFromImage(lab) > 0).sum()) != n_before:
                raise RuntimeError("reorienting the label changed its voxel count")
            reoriented = True
        la = sitk.GetArrayFromImage(lab)
        values = {int(v): int(c) for v, c in zip(*np.unique(la, return_counts=True))}
        mask = (la > 0).astype(np.uint8)
        if mask.sum() == 0:
            raise RuntimeError("empty label")
        raw_vol = float(mask.sum()) * float(np.prod(img.GetSpacing())) / 1000.0
        mimg = sitk.GetImageFromArray(mask); mimg.CopyInformation(img)
        img_rs = P.resample_to_spacing(P.reorient_lps(img), SPACING, sitk.sitkLinear, HU_AIR)
        msk_rs = P.resample_to_spacing(P.reorient_lps(mimg), SPACING, sitk.sitkNearestNeighbor, 0)
        ia = np.clip(sitk.GetArrayFromImage(img_rs), HU_MIN, HU_MAX).astype(np.int16)
        ma = (sitk.GetArrayFromImage(msk_rs) > 0).astype(np.uint8)
        if ia.shape != ma.shape:
            ma = ma[tuple(slice(0, s) for s in ia.shape)]
        if ma.sum() == 0:
            raise RuntimeError("empty mask after resampling")
        nz = np.argwhere(ma > 0)
        lo = np.maximum(nz.min(0) - MARGIN, 0); hi = np.minimum(nz.max(0) + MARGIN + 1, ma.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        full_shape = list(ia.shape)
        ia, ma = np.ascontiguousarray(ia[sl]), np.ascontiguousarray(ma[sl])
        np.save(out / f"{cid}_img.npy", ia); np.save(out / f"{cid}_msk.npy", ma)
        montage(ia, ma, out / "qc" / f"{cid}.png", cid)
        lb, ncomp = ndimage.label(ma)
        sizes = np.bincount(lb.ravel())[1:]
        row = dict(id=cid, case=pid, patient=pid, shape=list(ia.shape), tumour_cm3=float(ma.sum()) / 1000.0, tumour_cm3_native=round(raw_vol, 3),
                   n_components=int(ncomp), largest_component_frac=float(sizes.max() / ma.sum()),
                   outside_largest_frac=float(1.0 - sizes.max() / ma.sum()), n_components_ge100=int((sizes >= 100).sum()),
                   label_values=values, n_series_in_folder=len(cands), native_spacing=[round(float(s), 4) for s in img.GetSpacing()],
                   native_size=list(img.GetSize()), resampled_shape=full_shape,
                   hu_in_mask_mean=float(ia[ma > 0].mean()), label_reoriented=reoriented, **meta)
        json.dump(row, open(rp, "w"))
        return row
    except Exception as e:  # noqa: BLE001
        return {"id": cid, "case": pid, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}


def ctfm_uids():
    """Series UIDs of the CT-FM pre-training scan list (constants read by the disassembler; never unpickled)."""
    import pickletools, re
    uids = set()
    with open(ROOT / "data" / "external" / "ctfm_pretraining" / "ctfm_scan_list.pkl", "rb") as f:
        for op, arg, pos in pickletools.genops(f):
            if isinstance(arg, str) and arg.startswith("1.") and re.fullmatch(r"\d+(\.\d+)+", arg):
                uids.add(arg)
    return uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort", choices=["ASLS26", "ESLS26"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = OUT / a.cohort
    for d in (out, out / "rows", out / "qc"):
        d.mkdir(parents=True, exist_ok=True)
    pids = sorted(p for p in os.listdir(SRC / a.cohort / a.cohort) if (SRC / a.cohort / a.cohort / p).is_dir())
    if a.limit:
        pids = pids[:a.limit]
    rows, fails = [], []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, [(a.cohort, p) for p in pids]), 1):
            (fails if "error" in r else rows).append(r)
            if "error" in r:
                print(f"FAIL {r['case']}: {r['error']}", flush=True)
            if i % 25 == 0:
                print(f"{i}/{len(pids)} ok={len(rows)} fail={len(fails)}", flush=True)
    rows.sort(key=lambda r: r["id"]); fails.sort(key=lambda r: r["id"])
    json.dump(rows, open(out / "manifest.json", "w"), indent=1)
    json.dump(fails, open(out / "failures.json", "w"), indent=1)
    if not rows:
        print(f"no case of {a.cohort} passed; see failures.json"); return
    uids = ctfm_uids()
    hit = [r["id"] for r in rows if r["series_uid"] in uids or r["study_uid"] in uids]
    vols = np.array([r["tumour_cm3"] for r in rows])
    summ = dict(cohort=a.cohort, n_folders=len(pids), n_ok=len(rows), n_fail=len(fails), ctfm_scan_list_size=len(uids), n_series_in_ctfm_scan_list=len(hit),
                tumour_cm3_median=float(np.median(vols)), tumour_cm3_iqr=[float(np.percentile(vols, 25)), float(np.percentile(vols, 75))],
                tumour_cm3_min=float(vols.min()), tumour_cm3_max=float(vols.max()), frac_lt1cm3=float((vols < 1).mean()), frac_gt50cm3=float((vols > 50).mean()),
                frac_masks_multi_ge100=float(np.mean([r["n_components_ge100"] > 1 for r in rows])),
                outside_largest_mean=float(np.mean([r["outside_largest_frac"] for r in rows])),
                crop_shape_median=[float(x) for x in np.median(np.array([r["shape"] for r in rows]), 0)],
                cache_gb=float(sum(np.prod(r["shape"]) * 3 for r in rows) / 1e9),
                n_label_reoriented=int(sum(bool(r.get("label_reoriented")) for r in rows)))
    for k in ("manufacturer", "model", "kernel", "kvp", "slice_thickness", "contrast", "patient_position", "label_values"):
        c = {}
        for r in rows:
            v = json.dumps(r[k], sort_keys=True) if k == "label_values" else r[k]
            v = "values " + ",".join(sorted(json.loads(v))) if k == "label_values" else v
            c[v] = c.get(v, 0) + 1
        summ[k] = dict(sorted(c.items(), key=lambda kv: -kv[1])[:12])
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
