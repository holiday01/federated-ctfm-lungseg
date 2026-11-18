#!/usr/bin/env python3
"""Job lists for the first journal revision of Study 3 (JIIM, decision received 2026-11-16).

  jobs_r1_extended.json   E1  18,000 iterations per site for centralized pooling and FedAvg (both
                              initializations, seeds 7/42/1337). FedAvg keeps E = 200 and runs 90 rounds;
                              validation is evaluated at 18 (centralized) / 15 (FedAvg) points, as in
                              the 6,000-iteration runs.
  jobs_r1_fedbn_loso.json E4  FedBN on two sites with the third held out (CT-FM, 3 seeds); the held-out
                              normalization rules are compared afterwards by eval_fedbn_heldout.py.
  jobs_r1_fedprox.json    E3  FedProx with mu = 0.001 and 0.1 (mu = 0.01 exists), CT-FM, 3 seeds.
  jobs_r1_fixsplit.json   E2  fixed-split replications: split of seed 7, model seeds 11/23/101/211
                              (the existing seed-7 runs are the fifth replicate); local x3, centralized
                              all, FedAvg all; both initializations. Written to results/fl_fixsplit so that
                              aggregate_fl.py never merges them with the main grid.
All runs use lr 1e-4, the rate of the main grid (results/fl/lr_probe.json).
"""
import json
from pathlib import Path

ROOT = Path("/home/holiday/lung_ct")
PY = str(ROOT / ".venv" / "bin" / "python")
TRAIN = str(ROOT / "fl_study" / "train_fl.py")
OUT = Path(__file__).resolve().parent
SEEDS = [7, 42, 1337]
SITES = ["TW", "NL", "US"]
ALL = "TW,NL,US"
LR = 1e-4


def job(tag, *args):
    return {"tag": tag, "cmd": [PY, TRAIN, "--tag", tag, "--lr", str(LR), *map(str, args)]}


def extended():
    out = []
    for seed in SEEDS:
        for init in ("ctfm", "scratch"):
            out.append(job(f"budget18000_central_TW-NL-US_{init}_s{seed}", "--mode", "central", "--sites", ALL, "--init", init,
                           "--seed", seed, "--iters-per-site", 18000, "--eval-every", 3000))
            out.append(job(f"budget18000_fedavg_TW-NL-US_{init}_s{seed}", "--mode", "fedavg", "--sites", ALL, "--init", init,
                           "--seed", seed, "--iters-per-site", 18000, "--rounds", 90, "--local-iters", 200, "--eval-every-rounds", 6))
    return out


def fedbn_loso():
    out = []
    for seed in SEEDS:
        for held in SITES:
            pair = [s for s in SITES if s != held]
            out.append(job(f"fedbn_{'-'.join(pair)}_ctfm_s{seed}", "--mode", "fedavg", "--sites", ",".join(pair), "--init", "ctfm",
                           "--seed", seed, "--iters-per-site", 6000, "--rounds", 30, "--local-iters", 200, "--fedbn"))
    return out


def fedprox():
    out = []
    for seed in SEEDS:
        for mu in (0.001, 0.1):
            out.append(job(f"fedprox{mu:g}_TW-NL-US_ctfm_s{seed}", "--mode", "fedavg", "--sites", ALL, "--init", "ctfm",
                           "--seed", seed, "--iters-per-site", 6000, "--rounds", 30, "--local-iters", 200, "--fedprox-mu", mu))
    return out


def fixsplit(split_seed=7, model_seeds=(11, 23, 101, 211)):
    out = []
    budget = ["--iters-per-site", 6000, "--rounds", 30, "--local-iters", 200, "--split-seed", split_seed]
    for seed in model_seeds:
        for init in ("ctfm", "scratch"):
            for s in SITES:
                out.append(job(f"fs{split_seed}_local_{s}_{init}_s{seed}", "--mode", "local", "--sites", s, "--init", init,
                               "--seed", seed, "--no-heldout-eval", *budget))
            out.append(job(f"fs{split_seed}_central_TW-NL-US_{init}_s{seed}", "--mode", "central", "--sites", ALL, "--init", init,
                           "--seed", seed, *budget))
            out.append(job(f"fs{split_seed}_fedavg_TW-NL-US_{init}_s{seed}", "--mode", "fedavg", "--sites", ALL, "--init", init,
                           "--seed", seed, *budget))
    return out


if __name__ == "__main__":
    for name, jobs in (("extended", extended()), ("fedbn_loso", fedbn_loso()), ("fedprox", fedprox()), ("fixsplit", fixsplit())):
        json.dump(jobs, open(OUT / f"jobs_r1_{name}.json", "w"), indent=1)
        print(name, len(jobs), "jobs")
