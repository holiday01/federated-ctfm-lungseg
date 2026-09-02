#!/usr/bin/env python3
"""Download the segmentation-usable subset of TCIA 'NSCLC Radiogenomics' (Bakr et al. 2018)
through the public NBIA REST API (anonymous access, CC BY 3.0).

Stage 1 (--stage seg): download the 144 DICOM SEG series (one per patient, primary tumour).
Stage 2 (--stage ct):  parse each SEG's ReferencedSeriesSequence and download exactly the CT
                       series it was drawn on (nothing else: no PET, scouts, reformats).

Layout: data/external/nsclc_radiogenomics/dicom/<PatientID>/{SEG,CT}/<SeriesInstanceUID>/*.dcm
Resumable: a series is skipped when its directory holds a DONE marker.
"""
import argparse, io, json, sys, time, zipfile, concurrent.futures as cf
from pathlib import Path
import urllib.request, urllib.parse

ROOT = Path("/home/holiday/lung_ct")
OUT = ROOT / "data" / "external" / "nsclc_radiogenomics"
DCM = OUT / "dicom"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "NSCLC Radiogenomics"


def get(url, retries=5, timeout=900):
    last = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"download failed after {retries} tries: {url} ({last})")


def series_list():
    p = OUT / "series_index.json"
    if p.exists():
        return json.load(open(p))
    data = json.loads(get(f"{API}/getSeries?Collection={urllib.parse.quote(COLLECTION)}"))
    json.dump(data, open(p, "w"), indent=1)
    return data


def fetch_series(uid, dest: Path):
    if (dest / "DONE").exists():
        return "skip"
    dest.mkdir(parents=True, exist_ok=True)
    blob = get(f"{API}/getImage?SeriesInstanceUID={uid}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest)
    n = len(list(dest.glob("*.dcm")))
    (dest / "DONE").write_text(f"{n} files\n")
    return f"ok {n} files {len(blob)/1e6:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["seg", "ct"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    s = series_list()
    segs = [x for x in s if x["Modality"] == "SEG"]
    by_uid = {x["SeriesInstanceUID"]: x for x in s}
    print(f"{len(segs)} SEG series in index", flush=True)
    jobs = []
    if a.stage == "seg":
        for x in segs:
            jobs.append((x["PatientID"], x["SeriesInstanceUID"], DCM / x["PatientID"] / "SEG" / x["SeriesInstanceUID"]))
    else:
        import pydicom
        for x in segs:
            segdir = DCM / x["PatientID"] / "SEG" / x["SeriesInstanceUID"]
            fs = list(segdir.glob("*.dcm"))
            if not fs:
                print(f"{x['PatientID']}: SEG not downloaded yet, skip", flush=True)
                continue
            ds = pydicom.dcmread(fs[0], stop_before_pixels=True)
            ref = ds.ReferencedSeriesSequence[0].SeriesInstanceUID
            if ref not in by_uid:
                print(f"{x['PatientID']}: referenced CT {ref} not in collection index!", flush=True)
            jobs.append((x["PatientID"], ref, DCM / x["PatientID"] / "CT" / ref))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} {a.stage} series to fetch with {a.workers} workers", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_series, uid, dest): (pid, uid) for pid, uid, dest in jobs}
        for i, f in enumerate(cf.as_completed(futs)):
            pid, uid = futs[f]
            try:
                print(f"[{i+1}/{len(jobs)} {time.time()-t0:.0f}s] {pid} {f.result()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{i+1}/{len(jobs)}] {pid} FAILED {e}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
