#!/usr/bin/env python3
"""Queue runner for Study 3: runs train_fl.py jobs with N concurrent GPU workers.

A job is skipped when results/fl/<tag>.json already exists (resumable). Each job logs to
logs/fl/<tag>.log. The job list is produced by --grid {probe,main,ablation} or --jobs <file.json>.
"""
import argparse, json, os, subprocess, sys, time, threading, queue
from pathlib import Path

ROOT = Path("/home/holiday/lung_ct")
PY = str(ROOT / ".venv" / "bin" / "python")
TRAIN = str(ROOT / "fl_study" / "train_fl.py")
RESULTS = ROOT / "results" / "fl"
LOGS = ROOT / "logs" / "fl"
SEEDS = [7, 42, 1337]
SITES = ["TW", "NL", "US"]
ITERS = int(os.environ.get("FL_ITERS_PER_SITE", "6000"))   # gradient steps per participating site
LOCAL_E = 200                                                 # FL local iterations per round
ROUNDS = ITERS // LOCAL_E
BUDGET = ["--iters-per-site", ITERS, "--rounds", ROUNDS, "--local-iters", LOCAL_E]


def job(tag, *args):
    return {"tag": tag, "cmd": [PY, TRAIN, "--tag", tag, *map(str, BUDGET), *map(str, args)]}


def grid_probe(lrs=(1e-4, 3e-4, 1e-3)):
    out = []
    for init in ("ctfm", "scratch"):
        for lr in lrs:
            out.append(job(f"probe_local_TW_{init}_lr{lr:g}_s7", "--mode", "local", "--sites", "TW", "--init", init,
                           "--seed", 7, "--lr", lr, "--no-heldout-eval"))
    return out


def grid_main(inits=("ctfm", "scratch"), seeds=SEEDS, lr=None):
    out = []
    for seed in seeds:
        for init in inits:
            extra = ["--lr", lr[init]] if lr else []
            for s in SITES:
                out.append(job(f"local_{s}_{init}_s{seed}", "--mode", "local", "--sites", s, "--init", init, "--seed", seed, *extra))
            for s in SITES:
                others = ",".join(x for x in SITES if x != s)
                out.append(job(f"central_{others.replace(',', '-')}_{init}_s{seed}", "--mode", "central", "--sites", others,
                               "--init", init, "--seed", seed, *extra))
                out.append(job(f"fedavg_{others.replace(',', '-')}_{init}_s{seed}", "--mode", "fedavg", "--sites", others,
                               "--init", init, "--seed", seed, *extra))
            out.append(job(f"central_TW-NL-US_{init}_s{seed}", "--mode", "central", "--sites", "TW,NL,US", "--init", init, "--seed", seed, *extra))
            out.append(job(f"fedavg_TW-NL-US_{init}_s{seed}", "--mode", "fedavg", "--sites", "TW,NL,US", "--init", init, "--seed", seed,
                           "--personalize-iters", 500, *extra))
    return out


def grid_ablation(seeds=SEEDS, lr=None, init="ctfm"):
    out = []
    extra = ["--lr", lr[init]] if lr else []
    for seed in seeds:
        out.append(job(f"fedprox0.01_TW-NL-US_{init}_s{seed}", "--mode", "fedavg", "--sites", "TW,NL,US", "--init", init, "--seed", seed,
                       "--fedprox-mu", 0.01, *extra))
        out.append(job(f"fedavg-uniform_TW-NL-US_{init}_s{seed}", "--mode", "fedavg", "--sites", "TW,NL,US", "--init", init, "--seed", seed,
                       "--client-weighting", "uniform", *extra))
        out.append(job(f"fedavg-r{2*ROUNDS}e100_TW-NL-US_{init}_s{seed}", "--mode", "fedavg", "--sites", "TW,NL,US", "--init", init, "--seed", seed,
                       "--rounds", 2 * ROUNDS, "--local-iters", 100, "--eval-every-rounds", 4, *extra))
    # budget sensitivity: 3x the per-site budget for local TW training, both inits, seed 7
    for init2 in ("ctfm", "scratch"):
        lr2 = ["--lr", lr[init2]] if lr else []
        out.append(job(f"budget{3*ITERS}_local_TW_{init2}_s7", "--mode", "local", "--sites", "TW", "--init", init2, "--seed", 7,
                       "--iters-per-site", 3 * ITERS, "--no-heldout-eval", *lr2))
    return out


def grid_fedbn(seeds=SEEDS, lr=None, init="ctfm"):
    """FedBN (BatchNorm kept local) with size weighting, all three sites, + 500-it personalisation
    so it is comparable to both fedavg_all and its fine-tuned variant."""
    extra = ["--lr", lr[init]] if lr else []
    return [job(f"fedbn_TW-NL-US_{init}_s{seed}", "--mode", "fedavg", "--sites", "TW,NL,US", "--init", init, "--seed", seed,
                "--fedbn", "--personalize-iters", 500, *extra) for seed in seeds]


def run_job(j):
    tag = j["tag"]
    if (RESULTS / f"{tag}.json").exists():
        return f"{tag}: exists, skipped"
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOGS / f"{tag}.log", "w") as f:
        rc = subprocess.call(j["cmd"], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return f"{tag}: rc={rc} {(time.time()-t0)/60:.1f} min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["probe", "main", "ablation", "fedbn"])
    ap.add_argument("--jobs", help="JSON file with a list of {tag, cmd}")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr-json", help="JSON {init: lr} chosen by the probe")
    ap.add_argument("--inits", default="ctfm,scratch")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    lr = json.load(open(a.lr_json)) if a.lr_json else None
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.jobs:
        jobs = json.load(open(a.jobs))
    elif a.grid == "probe":
        jobs = grid_probe()
    elif a.grid == "main":
        jobs = grid_main(inits=tuple(a.inits.split(",")), seeds=seeds, lr=lr)
    elif a.grid == "fedbn":
        jobs = grid_fedbn(seeds=seeds, lr=lr)
    else:
        jobs = grid_ablation(seeds=seeds, lr=lr)
    pending = [j for j in jobs if not (RESULTS / f"{j['tag']}.json").exists()]
    print(f"{len(jobs)} jobs, {len(pending)} pending, {a.workers} workers", flush=True)
    for j in pending:
        print("  ", j["tag"], " ".join(j["cmd"][3:]), flush=True)
    if a.dry:
        return
    q = queue.Queue()
    for j in pending:
        q.put(j)
    lock = threading.Lock()

    def worker(wid):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            with lock:
                print(f"[w{wid} {time.strftime('%H:%M:%S')}] start {j['tag']}", flush=True)
            msg = run_job(j)
            with lock:
                print(f"[w{wid} {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
