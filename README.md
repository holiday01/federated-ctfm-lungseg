# federated-ctfm-lungseg

Code for *Simulated federated fine-tuning of a CT foundation model for ROI-based lung tumor
segmentation across three heterogeneous cohorts*. Local, centralized and simulated federated
(FedAvg and ablations) fine-tuning of the CT-FM SegResNet (`project-lighter/ct_fm_segresnet`)
on three cohorts, with leave-one-site-out evaluation and patient-level paired statistics.

## Pipeline
1. `fl_study/download_radiogenomics.py` (TCIA NBIA API) and `preprocess_radiogenomics.py`; NSCLC-Radiomics
   and the in-house cohort follow the same preprocessing (1 mm isotropic, LPS, HU [-1000, 400],
   reference-annotation bounding box + 80 mm margin).
2. `build_site_cache.py` writes memory-mapped `data/fl_cache/<site>/` and `manifest.json` (id, patient, volume).
3. `train_fl.py --mode {local,central,fedavg} --sites TW,NL,US --init {ctfm,scratch} --seed 7` writes
   `results/fl/<tag>.json` with per-case Dice, HD95, conditional ECE, ROI overlap and the validation
   history; see `--help` for the ablation flags (`--fedprox-mu`, `--client-weighting`, `--fedbn`,
   `--reset-client-opt`, `--site-balanced`, `--personalize-iters`).
4. `run_grid.py --grid {probe,main,ablation,fedbn}` runs the whole experiment (2 concurrent GPU workers).
5. `aggregate_fl.py` writes `results/fl/summary_fl.json` and `REPORT.md`: every number reported in the
   paper (patient-level paired differences, bootstrap CIs, Holm-adjusted p-values) derives from this
   summary. `eval_postproc.py` re-computes stored metrics from checkpoints.

`splits/` holds the seeded patient-level split definitions (case ids; in-house ids are anonymized hashes).
The in-house images cannot be shared; the two public cohorts are on TCIA (NSCLC-Radiomics, NSCLC Radiogenomics).

## Note on paths
Scripts reference the authors' directory layout (`/home/holiday/lung_ct`, and a read-only source
mount for the in-house DICOM); adapt `ROOT` / `CACHE` at the top of each script.
