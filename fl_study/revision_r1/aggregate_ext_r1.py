#!/usr/bin/env python3
"""Revision 1, analysis J: test-only evaluation on the external cohorts (ASLS26, ESLS26).

Reads results/fl_r1/ext/<COHORT>/<tag>.json (written by ext/eval_ext.py) and writes
results/fl_r1/ext_summary.json (+ .md digest).

Every model of a group (setting x initialization) is one seed of the first submission; all seeds
are evaluated on the same external patients, so paired comparisons average the seeds within a
patient and then resample patients (the procedure of aggregate_roi_r1.paired).

Each cohort has its own bootstrap stream and its own Holm families, so the numbers of one cohort do
not move when the other cohort is added. Within a cohort the order of the comparisons is fixed;
append new comparisons at the end of main().
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "fl_study")); sys.path.insert(0, str(ROOT / "fl_study" / "revision_r1"))
import aggregate_fl as A  # noqa: E402
from aggregate_roi_r1 import paired, seed_stat  # noqa: E402

R1 = ROOT / "results" / "fl_r1"
OUT = R1 / "ext_summary.json"
COHORTS = {"ASLS26": 20261118, "ESLS26": 20261119}          # cohort -> bootstrap seed
SETTINGS = ["local_TW", "local_NL", "local_US", "central_all", "fedavg_all"]
INITS = ["ctfm", "scratch"]
METRICS = ["dice", "dice_lcc", "recall", "precision"]
BANDS = [("<1", 0, 1), ("1-10", 1, 10), ("10-50", 10, 50), (">=50", 50, 1e9)]   # cm3, as in aggregate_fl.py
N_SEEDS = 3


def setting_of(r):
    sites = list(r["train_sites"])
    if r["mode"] == "local" or len(sites) == 1:
        return "local_" + sites[0]
    if len(sites) == 3 and r["variant"] in ("central", "fedavg"):
        return r["variant"] + "_all"
    return None


def load(cohort):
    G = defaultdict(dict)                                    # (setting, init) -> seed -> results
    for f in sorted(glob.glob(str(R1 / "ext" / cohort / "*.json"))):
        r = json.load(open(f)); st = setting_of(r)
        if st in SETTINGS and r["iters_per_site"] == 6000:
            G[(st, r["init"])][r["seed"]] = r["results"]
    return G


def tab(runs, metric, keep=None):
    """{seed: {case: value}}; keep = optional set of case ids."""
    return {s: {c: float(v) for c, v in zip(x["ids"], x[metric]) if keep is None or c in keep} for s, x in runs.items()}


def summarize(cohort, G, seed):
    rng = np.random.RandomState(seed)
    full = {k: v for k, v in G.items() if len(v) == N_SEEDS}
    out = {"n_models": int(sum(len(v) for v in G.values())), "incomplete": sorted(f"{k[0]}|{k[1]} ({len(v)} seeds)" for k, v in G.items() if len(v) != N_SEEDS),
           "groups": {}, "paired": {}, "bands": {}}
    if not full:
        return out
    any_run = next(iter(next(iter(full.values())).values()))
    ids = list(any_run["ids"]); vol = dict(zip(ids, any_run["gt_cm3"]))
    out["n_cases"] = len(ids)
    for (st, init), runs in sorted(full.items(), key=lambda kv: (SETTINGS.index(kv[0][0]), INITS.index(kv[0][1]))):
        g = {m: seed_stat(tab(runs, m)) for m in METRICS}
        g["detected"] = seed_stat({s: {c: float(v) for c, v in zip(x["ids"], x["detected"])} for s, x in runs.items()})
        comps = np.concatenate([x["n_pred_components"] for x in runs.values()])
        ratio = np.concatenate([np.array(x["pred_cm3"]) / np.maximum(np.array(x["gt_cm3"]), 1e-6) for x in runs.values()])
        g["n_pred_components_median"] = float(np.median(comps)); g["pred_over_ref_volume_median"] = float(np.median(ratio))
        g["seeds"] = sorted(runs)
        out["groups"][f"{st}|{init}"] = g
    fam = defaultdict(dict)

    def add(key, family, p):
        if p:
            out["paired"][key] = p; fam[family][key] = p

    # 1. initialization effect, one family per metric
    for m in METRICS:
        for st in SETTINGS:
            if (st, "ctfm") in full and (st, "scratch") in full:
                add(f"init|{m}|{st}", f"ext_{cohort}_init_{m}", paired(tab(full[(st, "ctfm")], m), tab(full[(st, "scratch")], m), cohort, rng))
    # 2. training settings against each other, Dice
    for init in INITS:
        pairs = [("fedavg_all", "central_all")] + [(a, f"local_{s}") for a in ("central_all", "fedavg_all") for s in ("TW", "NL", "US")]
        for a, b in pairs:
            if (a, init) in full and (b, init) in full:
                add(f"{a}-vs-{b}|dice|{init}", f"ext_{cohort}_settings_{init}", paired(tab(full[(a, init)], "dice"), tab(full[(b, init)], "dice"), cohort, rng))
    # 3. initialization effect on Dice by reference volume (descriptive; own family)
    for b, lo, hi in BANDS:
        keep = {c for c in ids if lo <= vol[c] < hi}
        out["bands"][b] = {"n_cases": len(keep), "dice": {}}
        if len(keep) < 5:
            continue
        for (st, init), runs in full.items():
            out["bands"][b]["dice"][f"{st}|{init}"] = seed_stat(tab(runs, "dice", keep))
        for st in SETTINGS:
            if (st, "ctfm") in full and (st, "scratch") in full:
                add(f"init_band|{b}|{st}", f"ext_{cohort}_init_bands", paired(tab(full[(st, "ctfm")], "dice", keep), tab(full[(st, "scratch")], "dice", keep), cohort, rng))
    for family, d in fam.items():
        adj = A.holm({k: v["wilcoxon_p"] for k, v in d.items()})
        for k, a in adj.items():
            d[k]["wilcoxon_p_holm"] = a; d[k]["holm_family"] = family; d[k]["holm_family_size"] = len(adj)
    return out


def main():
    S = {}
    for cohort, seed in COHORTS.items():
        G = load(cohort)
        if G:
            S[cohort] = summarize(cohort, G, seed)
    json.dump(S, open(OUT, "w"), indent=1)
    L = ["# Revision 1, J: external test-only cohorts", ""]
    f = lambda e: f"{e['mean']:.3f} +/- {e['sd']:.3f}"
    for cohort, s in S.items():
        L += [f"## {cohort}: {s.get('n_cases', 0)} patients, {s['n_models']} models; incomplete groups: {', '.join(s['incomplete']) or 'none'}", "",
              "| setting | init | Dice | Dice LCC | recall | precision | detected | pred. comp. (median) | pred/ref volume (median) |", "|---|---|---|---|---|---|---|---|---|"]
        for k, g in s["groups"].items():
            st, init = k.split("|")
            L.append(f"| {st} | {init} | {f(g['dice'])} | {f(g['dice_lcc'])} | {f(g['recall'])} | {f(g['precision'])} | {g['detected']['mean']:.3f} | {g['n_pred_components_median']:.0f} | {g['pred_over_ref_volume_median']:.2f} |")
        L += ["", "| comparison | n | diff | 95% CI | p | p_holm | family (tests) | seed diffs |", "|---|---|---|---|---|---|---|---|"]
        for k, p in s["paired"].items():
            L.append(f"| {k} | {p['n_patients']} | {p['mean_diff']:+.3f} | [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['wilcoxon_p']:.2g} | {p['wilcoxon_p_holm']:.2g} | {p['holm_family']} ({p['holm_family_size']}) | "
                     + ", ".join(f"{d:+.3f}" for d in p["seed_mean_diffs"]) + " |")
        L += ["", "| volume band (cm3) | n | " + " | ".join(s["groups"]) + " |", "|---|---|" + "---|" * len(s["groups"])]
        for b, e in s["bands"].items():
            L.append(f"| {b} | {e['n_cases']} | " + " | ".join(f"{e['dice'][k]['mean']:.3f}" if k in e["dice"] else "--" for k in s["groups"]) + " |")
        L.append("")
    open(OUT.with_suffix(".md"), "w").write("\n".join(L) + "\n")
    print("wrote", OUT, "| cohorts:", ", ".join(f"{c} ({s['n_models']} models)" for c, s in S.items()) or "none")


if __name__ == "__main__":
    main()
