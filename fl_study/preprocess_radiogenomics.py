#!/usr/bin/env python3
"""Preprocess TCIA 'NSCLC Radiogenomics' (site US) into the SAME format as the other two sites.

For each patient with a downloaded DICOM SEG and its referenced CT series:
  1. read the CT series (SimpleITK), read the SEG (pydicom, bit-unpacked frames),
  2. place every SEG frame on the CT slice whose ImagePositionPatient z matches (this collection's
     SEG objects carry no per-frame source-instance references, so slice position is the link),
  3. reorient to LPS, resample image (linear, air fill) and mask (nearest) to 1 mm isotropic,
     clip HU to [-1000, 400], crop to tumour bbox + 80 mm margin  -- identical to
     scripts/preprocess_seg.py and ssl_study/preprocess_external_nsclc.py,
  4. save data/external_radiogenomics_processed_1mm/rg_<pid>.npz {image:int16, mask:uint8, case:str}
     and a QC montage PNG (axial + coronal through the tumour centre) for visual review.

Acquisition metadata (no PHI; TCIA identifiers are public pseudo-IDs) goes to manifest_rg.csv.
"""
import csv, glob, json, sys, traceback
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "scripts"))
import preprocess as P  # noqa: E402

DCM = ROOT / "data" / "external" / "nsclc_radiogenomics" / "dicom"
OUT = ROOT / "data" / "external_radiogenomics_processed_1mm"
QC = ROOT / "results" / "fl" / "qc_rg"
SPACING = (1.0, 1.0, 1.0)
HU_MIN, HU_MAX = -1000, 400
HU_AIR = getattr(P, "HU_AIR", -1024)
MARGIN_MM = 80.0


def bbox_with_margin(mask, margin_vox):
    nz = np.argwhere(mask > 0)
    lo = np.maximum(nz.min(0) - margin_vox, 0)
    hi = np.minimum(nz.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def read_ct(ctdir):
    rdr = sitk.ImageSeriesReader()
    files = rdr.GetGDCMSeriesFileNames(str(ctdir))
    if not files:
        raise RuntimeError("no CT series found")
    rdr.SetFileNames(files)
    img = rdr.Execute()
    hdr = pydicom.dcmread(files[0], stop_before_pixels=True)
    meta = dict(manufacturer=str(getattr(hdr, "Manufacturer", "")), model=str(getattr(hdr, "ManufacturerModelName", "")),
                kernel=str(getattr(hdr, "ConvolutionKernel", "")), kvp=str(getattr(hdr, "KVP", "")),
                slice_thickness=str(getattr(hdr, "SliceThickness", "")), contrast=str(getattr(hdr, "ContrastBolusAgent", "")),
                series_desc=str(getattr(hdr, "SeriesDescription", "")))
    return img, meta


def seg_to_mask(segfile, img):
    ds = pydicom.dcmread(segfile)
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    nz = img.GetSize()[2]
    zs = np.array([img.TransformIndexToPhysicalPoint((0, 0, k))[2] for k in range(nz)])
    dz = float(abs(zs[1] - zs[0])) if nz > 1 else 1.0
    arr_shape = (nz, img.GetSize()[1], img.GetSize()[0])
    if frames.shape[1:] != arr_shape[1:]:
        raise RuntimeError(f"SEG rows/cols {frames.shape[1:]} != CT {arr_shape[1:]}")
    # orientation check: SEG frames must share the CT's in-plane orientation
    sfg = ds.SharedFunctionalGroupsSequence[0]
    iop = None
    if hasattr(sfg, "PlaneOrientationSequence"):
        iop = [float(v) for v in sfg.PlaneOrientationSequence[0].ImageOrientationPatient]
    elif hasattr(ds.PerFrameFunctionalGroupsSequence[0], "PlaneOrientationSequence"):
        iop = [float(v) for v in ds.PerFrameFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient]
    d = np.array(img.GetDirection()).reshape(3, 3)
    ct_iop = list(d[:, 0]) + list(d[:, 1])
    if iop is not None and np.max(np.abs(np.array(iop) - np.array(ct_iop))) > 1e-3:
        raise RuntimeError(f"SEG orientation {np.round(iop, 3)} != CT {np.round(ct_iop, 3)}")
    mask = np.zeros(arr_shape, np.uint8)
    unmatched = 0
    for i, pf in enumerate(ds.PerFrameFunctionalGroupsSequence):
        z = float(pf.PlanePositionSequence[0].ImagePositionPatient[2])
        k = int(np.argmin(np.abs(zs - z)))
        if abs(zs[k] - z) > dz / 2 + 1e-3:
            unmatched += 1
            continue
        mask[k] |= frames[i].astype(np.uint8)
    labels = [str(s.SegmentLabel) for s in ds.SegmentSequence]
    return mask, unmatched, labels


def montage(ia, ma, spacing_zyx, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zc = int(np.median(np.where(ma)[0])); yc = int(np.median(np.where(ma)[1]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(ia[zc], cmap="gray", vmin=-1000, vmax=400); ax[0].contour(ma[zc], colors="r", linewidths=0.7)
    ax[0].set_title(f"{title} axial"); ax[0].axis("off")
    ax[1].imshow(ia[:, yc, :], cmap="gray", vmin=-1000, vmax=400, aspect=spacing_zyx[0] / spacing_zyx[2], origin="lower")
    ax[1].contour(ma[:, yc, :], colors="r", linewidths=0.7, origin="lower"); ax[1].set_title("coronal"); ax[1].axis("off")
    plt.savefig(path, dpi=70, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); QC.mkdir(parents=True, exist_ok=True)
    margin_vox = int(round(MARGIN_MM / SPACING[0]))
    rows, n_ok, n_fail = [], 0, 0
    for pdir in sorted(DCM.iterdir()):
        pid = pdir.name
        outp = OUT / f"rg_{pid}.npz"
        segs = glob.glob(str(pdir / "SEG" / "*" / "*.dcm"))
        cts = [d for d in (pdir / "CT").glob("*") if (d / "DONE").exists()] if (pdir / "CT").exists() else []
        if not segs or not cts:
            print(f"{pid}: waiting (seg={len(segs)} ct_done={len(cts)})", flush=True)
            continue
        if outp.exists():
            n_ok += 1
            continue
        try:
            img, meta = read_ct(cts[0])
            mask, unmatched, labels = seg_to_mask(segs[0], img)
            if mask.sum() == 0:
                raise RuntimeError("empty mask after alignment")
            raw_vol = float(mask.sum()) * float(np.prod(img.GetSpacing())) / 1000.0
            mimg = sitk.GetImageFromArray(mask); mimg.CopyInformation(img)
            img_l = P.reorient_lps(img); msk_l = P.reorient_lps(mimg)
            img_rs = P.resample_to_spacing(img_l, SPACING, sitk.sitkLinear, HU_AIR)
            msk_rs = P.resample_to_spacing(msk_l, SPACING, sitk.sitkNearestNeighbor, 0)
            ia = np.clip(sitk.GetArrayFromImage(img_rs), HU_MIN, HU_MAX).astype(np.int16)
            ma = (sitk.GetArrayFromImage(msk_rs) > 0).astype(np.uint8)
            if ia.shape != ma.shape:
                ma = ma[tuple(slice(0, s) for s in ia.shape)]
            if ma.sum() == 0:
                raise RuntimeError("empty mask after resample")
            sl = bbox_with_margin(ma, margin_vox)
            ia, ma = ia[sl], ma[sl]
            np.savez_compressed(outp, image=ia, mask=ma, case=pid)
            montage(ia, ma, (1, 1, 1), QC / f"{pid}.png", pid)
            vol = float(ma.sum()) * np.prod(SPACING) / 1000.0
            rows.append(dict(pid=pid, tumour_cm3=round(vol, 3), tumour_cm3_native=round(raw_vol, 3), unmatched_frames=unmatched,
                             seg_label="|".join(labels), native_spacing="x".join(f"{s:.3f}" for s in img.GetSpacing()),
                             native_size="x".join(str(s) for s in img.GetSize()), crop_shape="x".join(str(s) for s in ia.shape), **meta))
            n_ok += 1
            print(f"{pid}: ok vol={vol:.1f}cm3 native={raw_vol:.1f} unmatched={unmatched} spacing={meta['slice_thickness']} {meta['manufacturer']} {meta['kernel']}", flush=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"{pid}: FAIL {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    mpath = OUT / "manifest_rg.csv"
    old = []
    if mpath.exists():
        old = list(csv.DictReader(open(mpath)))
    done = {r["pid"] for r in rows}
    allrows = [r for r in old if r["pid"] not in done] + rows
    if allrows:
        keys = sorted({k for r in allrows for k in r.keys()}, key=lambda k: (k != "pid", k))
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(sorted(allrows, key=lambda r: r["pid"]))
    print(f"done: ok={n_ok} fail={n_fail} manifest rows={len(allrows)}")


if __name__ == "__main__":
    main()
