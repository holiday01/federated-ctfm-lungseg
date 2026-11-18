# federated-ctfm-lungseg

Code for *Federated fine-tuning of a CT foundation model for ROI-based lung tumor
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

## Additional analyses of the first journal revision (`fl_study/revision_r1/`)
- Training queues: `make_jobs_r1.py` writes the job lists and `launch_r1.sh` runs them with `run_grid.py`:
  centralized pooling and FedAvg at three times the budget, FedBN on two sites with the third held out,
  five replications on the fixed split of seed 7, and FedProx with two further proximal weights.
- Inference only, on stored checkpoints: `eval_roi_r1.py` (smaller, shifted and rough ROIs, false-positive
  components, component filters, calibration in fixed regions; `launch_roi_r1.sh`, `tags_roi_*.txt`) and
  `eval_fedbn_heldout.py` (five normalization states for a site outside a FedBN federation).
- From existing runs: `analyze_existing_r1.py` (communication to a shared validation score, initialization
  effect by pre-training exposure) and `mask_structure_r1.py` (connected components of the reference masks).
- Test-only external cohorts ESLS26 and ASLS26 (Science Data Bank, doi:10.57760/sciencedb.29318):
  `ext/preprocess_ext.py`, `ext/eval_ext.py` with the model list `ext/tags_ext.txt`, and
  `ext/diag_ext_orient.py`, the check of the slice order of the ESLS26 labels.
- Aggregation: `aggregate_roi_r1.py`, `aggregate_r1.py`, `aggregate_ext_r1.py`. Every comparison of the
  revision has its own bootstrap stream and Holm family, so the numbers of `aggregate_fl.py` do not change.
- `preprocess_radiogenomics.py` and `ext/preprocess_ext.py` import two helpers, `reorient_lps` and
  `resample_to_spacing` (SimpleITK), from the authors' preprocessing module, which is not part of this release.

`splits/` holds the seeded patient-level split definitions (case ids; in-house ids are anonymized hashes).
The in-house images cannot be shared; the two public cohorts are on TCIA (NSCLC-Radiomics, NSCLC Radiogenomics).

## Note on paths
Scripts reference the authors' directory layout (`/home/holiday/lung_ct`, and a read-only source
mount for the in-house DICOM); adapt `ROOT` / `CACHE` at the top of each script.
