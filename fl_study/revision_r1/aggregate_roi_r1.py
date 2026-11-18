#!/usr/bin/env python3
"""Revision-1 aggregation of the inference-only analyses.

Reads results/fl_r1/roi/<tag>.json (eval_roi_r1.py) and writes results/fl_r1/roi_summary.json:

  C  less precise ROIs at inference (CT-FM local / centralized / FedAvg): Dice per site and box,
     the Dice of the ORIGINAL prediction restricted to the same box, and the paired differences
     box minus 80 mm reference and box minus restricted original.
  D  false-positive components of the raw prediction and four component filters (largest
     component; minimum size 0.01 / 0.1 / 1 cm3): Dice, HD95, tumor recall, share of correctly
     segmented tumor removed, cases in which the tumor was removed entirely; also by volume band.
  E  calibration error in the original prediction-dependent region and in two regions that are
     identical for every model (reference + 10 voxels; whole ROI).

Statistics follow aggregate_fl.py (seed-level mean +/- SD for description; patient-level paired
differences with a patient bootstrap and a Wilcoxon signed-rank test). This script has its OWN
bootstrap RNG and its OWN Holm families, so no number of the first submission changes.
Groups with fewer than three seeds are reported with the seeds they have and flagged.
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "fl_study"))
import aggregate_fl as A  # noqa: E402

ROI = ROOT / "results" / "fl_r1" / "roi"
OUT = ROOT / "results" / "fl_r1" / "roi_summary.json"
SITES = A.SITES
N_BOOT = 2000
CONDS = ["m60", "m40", "m20", "m40s20", "m40s40", "rough"]
FILTERS = ["lcc", "min10", "min100", "min1000"]
BANDS = [("<1", 0, 1), ("1-10", 1, 10), ("10-50", 10, 50), (">=50", 50, 1e9)]
EMPTY_FILTER = {"dice": 0.0, "recall": 0.0, "tp_removed_frac": 0.0, "tumor_lost": False, "hd95": float("nan")}


def seed_stat(T):
    """T = {seed: {case: value}} -> seed-level summary (NaN cases ignored within a seed)."""
    per = {s: float(np.nanmean(list(v.values()))) if len(v) else float("nan") for s, v in T.items()}
    m = np.array([x for x in per.values() if x == x])
    return {"n_seeds": int(len(m)), "mean": float(m.mean()) if len(m) else float("nan"),
            "sd": float(m.std(ddof=1)) if len(m) > 1 else float("nan"), "per_seed": {str(k): v for k, v in per.items()},
            "n_cases_per_seed": {str(k): len(v) for k, v in T.items()}}


def paired(TA, TB, site, rng):
    """Patient-level paired difference A - B. TA / TB = {seed: {case: value}} on identical cases."""
    by_case = defaultdict(list); seed_d = []
    for s in sorted(set(TA) & set(TB)):
        common = sorted(set(TA[s]) & set(TB[s]))
        d = [TA[s][c] - TB[s][c] for c in common if TA[s][c] == TA[s][c] and TB[s][c] == TB[s][c]]
        for c in common:
            if TA[s][c] == TA[s][c] and TB[s][c] == TB[s][c]:
                by_case[c].append(TA[s][c] - TB[s][c])
        if d:
            seed_d.append(float(np.mean(d)))
    if not by_case:
        return None
    by_pat = defaultdict(list)
    for c, v in by_case.items():
        by_pat[A.PATIENT.get(site, {}).get(c, c)].append(float(np.mean(v)))
    pat = np.array([np.mean(v) for v in by_pat.values()])
    try:
        p = float(stats.wilcoxon(pat).pvalue) if np.any(pat != 0) else 1.0
    except ValueError:
        p = float("nan")
    boots = [pat[rng.randint(len(pat), size=len(pat))].mean() for _ in range(N_BOOT)]
    return {"n_patients": int(len(pat)), "n_lesions": len(by_case), "n_evals": int(sum(len(v) for v in by_case.values())),
            "mean_diff": float(pat.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wilcoxon_p": p, "seed_mean_diffs": seed_d, "frac_patients_positive": float(np.mean(pat > 0)),
            "frac_patients_negative": float(np.mean(pat < 0))}


def tab(recs, site, fn):
    """{seed: {case: fn(case record)}} for one group and site."""
    return {s: {c["id"]: fn(c) for c in r["sites"][site]["cases"]} for s, r in recs.items() if site in r["sites"]}


def filt(c, name, key):
    return c["filters"].get(name, EMPTY_FILTER).get(key, float("nan")) if name != "raw" else c.get(key, float("nan"))


def load():
    by = defaultdict(dict)
    for f in sorted(glob.glob(str(ROI / "*.json"))):
        r = json.load(open(f))
        run = json.load(open(A.RES / f"{r['tag']}.json"))
        by[(A.setting_of(run), r["init"])][r["seed"]] = r
    return by


def main():
    rng = np.random.RandomState(20261116)
    by = load()
    S = {"n_files": int(sum(len(v) for v in by.values())), "groups": {f"{k[0]}|{k[1]}": sorted(v) for k, v in sorted(by.items())},
         "verify_max_abs_dice_diff": float(max(r["verify"]["max_abs_dice_diff"] for v in by.values() for r in v.values())),
         "perturb": {}, "fp": {}, "filters": {}, "filters_by_band": {}, "ece": {}, "ece_checks": {}}
    fam = defaultdict(dict)

    # ------------------------------------------------------------------ C: less precise ROIs
    for site in SITES:
        for label, setting in (("local", f"local_{site}"), ("central_all", "central_all"), ("fedavg_all", "fedavg_all")):
            recs = {s: r for s, r in by.get((setting, "ctfm"), {}).items() if r.get("perturb")}
            if not recs:
                continue
            base = tab(recs, site, lambda c: c["dice"])
            base_lcc = tab(recs, site, lambda c: filt(c, "lcc", "dice"))
            entry = {"seeds": sorted(recs), "base": {"dice": seed_stat(base), "dice_lcc": seed_stat(base_lcc),
                                                      "n_comp_median": float(np.median([c["n_comp"] for r in recs.values() for c in r["sites"][site]["cases"]]))}}
            for cond in CONDS:
                d = tab(recs, site, lambda c: c["perturb"][cond]["dice"])
                dl = tab(recs, site, lambda c: c["perturb"][cond]["dice_lcc"])
                inbox = tab(recs, site, lambda c: c["perturb"][cond]["dice_base_inbox"])
                cases = [c for r in recs.values() for c in r["sites"][site]["cases"]]
                e = {"dice": seed_stat(d), "dice_lcc": seed_stat(dl), "base_inbox": seed_stat(inbox),
                     "n_comp_median": float(np.median([c["perturb"][cond]["n_comp"] for c in cases])),
                     "min_margin_median_mm": float(np.median([min(c["perturb"][cond]["margin_lo"] + c["perturb"][cond]["margin_hi"]) for c in cases])),
                     "diff_vs_base": paired(d, base, site, rng), "diff_vs_inbox": paired(d, inbox, site, rng),
                     "diff_lcc_vs_base_lcc": paired(dl, base_lcc, site, rng)}
                key = f"{label}|{site}|{cond}"
                fam["roi_vs_base"][key] = e["diff_vs_base"]; fam["roi_vs_inbox"][key] = e["diff_vs_inbox"]
                fam["roi_lcc_vs_base_lcc"][key] = e["diff_lcc_vs_base_lcc"]
                entry[cond] = e
            S["perturb"][f"{label}|{site}"] = entry

    # ------------------------------------------------------------------ D: false-positive components and filters; E: calibration
    MAIN = {f"local_{s}" for s in SITES} | {"central_all", "fedavg_all"}
    for (setting, init), recs in sorted(by.items()):
        for site in SITES:
            if not any(site in r["sites"] for r in recs.values()):
                continue
            rs = {s: r for s, r in recs.items() if site in r["sites"]}
            cases = [c for r in rs.values() for c in r["sites"][site]["cases"]]
            key = f"{setting}|{init}|{site}"
            heldout = bool(next(iter(rs.values()))["sites"][site]["heldout"])
            S["fp"][key] = {
                "seeds": sorted(rs), "heldout": heldout, "n_evals": len(cases),
                "n_comp_median": float(np.median([c["n_comp"] for c in cases])),
                "n_fp_comp_median": float(np.median([c["n_fp_comp"] for c in cases])),
                "n_fp_comp_iqr": [float(np.percentile([c["n_fp_comp"] for c in cases], q)) for q in (25, 75)],
                "frac_cases_with_fp": seed_stat(tab(rs, site, lambda c: float(c["n_fp_comp"] > 0))),
                "frac_cases_tumor_touched": seed_stat(tab(rs, site, lambda c: float(c["n_tp_comp"] > 0))),
                "fp_cm3_median": float(np.median([c["fp_cm3"] for c in cases])),
                "largest_fp_cm3_median": float(np.median([c["largest_fp_cm3"] for c in cases])),
                "fp_frac_of_pred": seed_stat(tab(rs, site, lambda c: c["fp_frac_of_pred"])),
                "fp_dist_median_mm": float(np.nanmedian([c["fp_dist_median_mm"] for c in cases])) if any(c["n_fp_comp"] for c in cases) else float("nan"),
                "fp_vol_frac_ge20mm": seed_stat(tab(rs, site, lambda c: c["fp_vol_frac_ge20mm"])),
            }
            raw = tab(rs, site, lambda c: c["dice"])
            F = {"raw": {"dice": seed_stat(raw), "recall": seed_stat(tab(rs, site, lambda c: c["recall"])),
                         "hd95_median_mean": float(np.mean([np.nanmedian([c.get("hd95", float("nan")) for c in r["sites"][site]["cases"]]) for r in rs.values()])),
                         "empty_pred_frac": float(np.mean([c["n_comp"] == 0 for c in cases]))}}
            for name in FILTERS:
                d = tab(rs, site, lambda c: filt(c, name, "dice"))
                e = {"dice": seed_stat(d), "recall": seed_stat(tab(rs, site, lambda c: filt(c, name, "recall"))),
                     "tp_removed_frac": seed_stat(tab(rs, site, lambda c: filt(c, name, "tp_removed_frac"))),
                     "tp_removed_frac_max": float(np.max([filt(c, name, "tp_removed_frac") for c in cases])),
                     "frac_cases_tp_removed_gt10pct": float(np.mean([filt(c, name, "tp_removed_frac") > 0.10 for c in cases])),
                     "tumor_lost_evals": int(sum(bool(filt(c, name, "tumor_lost")) for c in cases)),
                     "tumor_lost_lesions": len({c["id"] for c in cases if filt(c, name, "tumor_lost")}),
                     "diff_vs_raw": paired(d, raw, site, rng)}
                if name != "min10":
                    e["hd95_median_mean"] = float(np.mean([np.nanmedian([filt(c, name, "hd95") for c in r["sites"][site]["cases"]]) for r in rs.values()]))
                    e["empty_after_filter_frac"] = float(np.mean([filt(c, name, "hd95") != filt(c, name, "hd95") for c in cases]))
                fam["filter_vs_raw_main" if (setting in MAIN and init == "ctfm") else "filter_vs_raw_other"][f"{key}|{name}"] = e["diff_vs_raw"]
                F[name] = e
            S["filters"][key] = F
            # by tumor-volume band: unit = unique lesion, values averaged over the seeds that evaluated it
            acc = defaultdict(lambda: defaultdict(list)); vol = {}
            for c in cases:
                vol[c["id"]] = c["gt_cm3"]
                for name in ["raw"] + FILTERS:
                    acc[name][c["id"]].append((filt(c, name, "dice"), filt(c, name, "tp_removed_frac") if name != "raw" else 0.0,
                                               bool(filt(c, name, "tumor_lost")) if name != "raw" else False))
            B = {}
            for b, lo, hi in BANDS:
                ids = [c for c in vol if lo <= vol[c] < hi]
                B[b] = {"n_lesions": len(ids), "n_patients": len({A.PATIENT.get(site, {}).get(c, c) for c in ids}),
                        "n_evals": int(sum(len(acc["raw"][c]) for c in ids))}
                for name in ["raw"] + FILTERS:
                    B[b][name] = {"dice": float(np.mean([np.mean([x[0] for x in acc[name][c]]) for c in ids])) if ids else None,
                                  "tp_removed_frac": float(np.mean([np.mean([x[1] for x in acc[name][c]]) for c in ids])) if ids else None,
                                  "tumor_lost_evals": int(sum(x[2] for c in ids for x in acc[name][c]))}
            S["filters_by_band"][key] = B
            S["ece"][key] = {"seeds": sorted(rs), "heldout": heldout,
                             "cond": seed_stat(tab(rs, site, lambda c: c["ece_cond"])),
                             "band": seed_stat(tab(rs, site, lambda c: c["ece_band"])),
                             "roi": seed_stat(tab(rs, site, lambda c: c["ece_roi"]))}

    # paired calibration differences in the common region, mirroring the Dice comparisons of the main text (CT-FM)
    for site in SITES:
        L, C, Fd = by.get((f"local_{site}", "ctfm"), {}), by.get(("central_all", "ctfm"), {}), by.get(("fedavg_all", "ctfm"), {})
        for name, (X, Y) in {"fedavg_all-vs-local": (Fd, L), "fedavg_all-vs-central_all": (Fd, C), "central_all-vs-local": (C, L)}.items():
            for region in ("cond", "band", "roi"):
                if X and Y:
                    res = paired(tab(X, site, lambda c: c[f"ece_{region}"]), tab(Y, site, lambda c: c[f"ece_{region}"]), site, rng)
                    if res:
                        S.setdefault("ece_paired", {})[f"{name}|{site}|{region}"] = res; fam[f"ece_{region}"][f"{name}|{site}|{region}"] = res

    # does the ordering of the first submission hold in the common regions?
    for region in ("cond", "band", "roi"):
        chk = {}
        for setting in sorted({k.split("|")[0] for k in S["ece"]}):
            for init in ("ctfm", "scratch"):
                v = {s: S["ece"].get(f"{setting}|{init}|{s}", {}).get(region, {}).get("mean") for s in SITES}
                if all(x is not None and x == x for x in v.values()) and not any(S["ece"][f"{setting}|{init}|{s}"]["heldout"] for s in SITES):
                    chk[f"{setting}|{init}"] = {"values": v, "largest_site": max(v, key=v.get)}
        for init in ("ctfm", "scratch"):
            v = {s: S["ece"].get(f"local_{s}|{init}|{s}", {}).get(region, {}).get("mean") for s in SITES}
            if all(x is not None for x in v.values()):
                chk[f"local_own|{init}"] = {"values": v, "largest_site": max(v, key=v.get)}
        S["ece_checks"][region] = {"per_setting": chk, "nl_largest_in_all": all(c["largest_site"] == "NL" for c in chk.values()), "n_settings": len(chk)}
    keys = [k for k in S["ece"] if not S["ece"][k]["heldout"]]
    for region in ("band", "roi"):
        a = [S["ece"][k]["cond"]["mean"] for k in keys]; b = [S["ece"][k][region]["mean"] for k in keys]
        rho = stats.spearmanr(a, b)
        S["ece_checks"][f"spearman_cond_vs_{region}"] = {"rho": float(rho.statistic), "n_cells": len(keys)}

    # ------------------------------------------------------------------ C (continued): setting comparisons under each altered box.
    # Computed last so that the bootstrap stream of everything above is unchanged. The 80 mm versions are in Table S3.
    S["perturb_settings"] = {}
    for site in SITES:
        G = {lab: {s: r for s, r in by.get((st, "ctfm"), {}).items() if r.get("perturb")}
             for lab, st in (("local", f"local_{site}"), ("central_all", "central_all"), ("fedavg_all", "fedavg_all"))}
        for a_, b_ in (("fedavg_all", "local"), ("fedavg_all", "central_all"), ("central_all", "local")):
            if not G[a_] or not G[b_]:
                continue
            for cond in CONDS:
                res = paired(tab(G[a_], site, lambda c: c["perturb"][cond]["dice"]), tab(G[b_], site, lambda c: c["perturb"][cond]["dice"]), site, rng)
                if res:
                    key = f"{a_}-vs-{b_}|{site}|{cond}"
                    S["perturb_settings"][key] = res; fam[f"roi_setting_{a_}-vs-{b_}"][key] = res

    for family, d in fam.items():
        adj = A.holm({k: v["wilcoxon_p"] for k, v in d.items() if v})
        for k, a in adj.items():
            d[k]["wilcoxon_p_holm"] = a; d[k]["holm_family"] = family; d[k]["holm_family_size"] = len(adj)
    json.dump(S, open(OUT, "w"), indent=1)
    print("wrote", OUT, "| files:", S["n_files"], "| groups:", len(by), "| max verify diff:", S["verify_max_abs_dice_diff"])

    # ------------------------------------------------------------------ digest for human review
    L = ["# Revision 1: inference-only analyses (C, D, E)", "", f"files: {S['n_files']}; groups with fewer than 3 seeds: "
         + (", ".join(k for k, v in S["groups"].items() if len(v) < 3) or "none"), "",
         "## C. Less precise ROIs (CT-FM; test Dice, mean over seeds; diff = box minus 80 mm reference, patient level)", "",
         "| model | site | 80 mm | " + " | ".join(CONDS) + " |", "|---|---|---|" + "---|" * len(CONDS)]
    for k, e in S["perturb"].items():
        L.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {e['base']['dice']['mean']:.3f} | "
                 + " | ".join(f"{e[c]['dice']['mean']:.3f} ({e[c]['diff_vs_base']['mean_diff']:+.3f}; inbox {e[c]['base_inbox']['mean']:.3f})" for c in CONDS) + " |")
    L += ["", "## D. False-positive components and filters (test Dice; tumor removed = mean share of reference volume that was correctly segmented and removed by the filter)", "",
          "| setting | init | site | seeds | FP comps median | cases with FP | FP share of pred | raw | LCC | min 0.01 | min 0.1 | min 1 cm3 | LCC tumor removed | LCC tumor lost (evals) | min1 tumor lost |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, F in S["filters"].items():
        st, init, site = k.split("|"); fp = S["fp"][k]
        L.append(f"| {st} | {init} | {site}{' (held out)' if fp['heldout'] else ''} | {len(fp['seeds'])} | {fp['n_fp_comp_median']:.0f} | {fp['frac_cases_with_fp']['mean']:.2f} | {fp['fp_frac_of_pred']['mean']:.2f} | "
                 + " | ".join(f"{F[n]['dice']['mean']:.3f}" for n in ["raw"] + FILTERS)
                 + f" | {F['lcc']['tp_removed_frac']['mean']:.3f} | {F['lcc']['tumor_lost_evals']}/{fp['n_evals']} | {F['min1000']['tumor_lost_evals']}/{fp['n_evals']} |")
    L += ["", "## E. Calibration error by region (mean over seeds)", "", "| setting | init | site | prediction-dependent (original) | reference + 10 voxels | whole ROI |", "|---|---|---|---|---|---|"]
    for k, e in S["ece"].items():
        st, init, site = k.split("|")
        L.append(f"| {st} | {init} | {site}{' (held out)' if e['heldout'] else ''} | {e['cond']['mean']:.4f} | {e['band']['mean']:.4f} | {e['roi']['mean']:.4f} |")
    L += ["", "NL largest in every in-federation setting: " + ", ".join(f"{r}: {S['ece_checks'][r]['nl_largest_in_all']} ({S['ece_checks'][r]['n_settings']} settings)" for r in ("cond", "band", "roi"))]
    open(OUT.with_suffix(".md"), "w").write("\n".join(L) + "\n")
    print("wrote", OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
