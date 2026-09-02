#!/usr/bin/env python3
"""Aggregate Study 3 runs (results/fl/*.json) into one provenance-verified summary.

Outputs
  results/fl/summary_fl.json     every number used downstream (tables, figures, manuscript)
  results/fl/REPORT.md           human-readable digest
  manuscripts/study3_fl/tables/*.tex   paste-ready LaTeX tables

Analysis units
  * seed level: per (setting, init, site) the per-seed mean Dice over that site's evaluation cases,
    then mean +/- SD and a 95% t-interval across seeds. Descriptive only.
  * patient level, paired (primary inference): two settings that share seed and init are compared
    on identical cases. Per-lesion differences are first averaged over the seeds that evaluated
    the lesion, lesions are then averaged within patient, giving ONE value per patient. The mean
    difference, the 95% bootstrap CI (resampling patients) and the Wilcoxon signed-rank test all
    operate on these patient values, so seeds are treated as repeated measures and multi-lesion
    patients as clusters -- never as independent observations. n_patients / n_lesions / n_evals
    (model-case evaluations) are all reported. Holm adjustment within each comparison family
    (one family = one comparison type across the three sites and, where applicable, both inits).
"""
from __future__ import annotations
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/holiday/lung_ct")
RES = ROOT / "results" / "fl"
TAB = ROOT / "manuscripts" / "study3_fl" / "tables"
CACHE = ROOT / "data" / "fl_cache"
SITES = ["TW", "NL", "US"]
SITE_NAMES = {"TW": "TW (in-house)", "NL": "NL (NSCLC-Radiomics)", "US": "US (NSCLC-Radiogenomics)"}
ALL = "TW-NL-US"
SEEDS = [7, 42, 1337]
N_BOOT = 2000
# case id -> patient id, per site (multi-lesion TW patients contribute several cases; NL/US are 1:1)
PATIENT = {site: {r["id"]: r["patient"] for r in json.load(open(CACHE / site / "manifest.json"))}
           for site in SITES if (CACHE / site / "manifest.json").exists()}


# ----------------------------------------------------------------------------- load
def load_runs():
    runs = []
    for f in sorted(glob.glob(str(RES / "*.json"))):
        name = Path(f).name
        if name.startswith(("probe_", "smoke_", "summary", "site_summary", "lr_")):
            continue
        r = json.load(open(f))
        if "results" not in r:
            continue
        runs.append(r)
    return runs


DEFAULT_ITERS, DEFAULT_ROUNDS, DEFAULT_E = 6000, 30, 200


def setting_of(r):
    """Canonical setting name independent of seed/init. Non-default budgets or federated
    schedules get an explicit suffix so ablation runs never merge with the main runs."""
    v, ts, cfg = r["variant"], "-".join(r["train_sites"]), r["config"]
    if r["mode"] == "fedavg" and (cfg.get("rounds"), cfg.get("local_iters")) != (DEFAULT_ROUNDS, DEFAULT_E):
        v += f"-r{cfg['rounds']}e{cfg['local_iters']}"
    if r["mode"] == "local":
        base = f"local_{ts}"
    elif ts == ALL:
        base = f"{v}_all"
    else:
        base = f"{v}_loso_{[s for s in SITES if s not in r['train_sites']][0]}"
    if cfg.get("iters_per_site", DEFAULT_ITERS) != DEFAULT_ITERS:
        base += f"_b{cfg['iters_per_site']}"
    return base


def case_table(r, site):
    """{split: {case_id: metrics}} for one run and site."""
    out = {}
    for split, res in r["results"].get(site, {}).items():
        out[split] = {cid: {k: res[k][i] for k in res if k != "ids"} for i, cid in enumerate(res["ids"])}
    return out


# ----------------------------------------------------------------------------- statistics
def t_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def seed_summary(vals_by_seed):
    m = np.array([np.mean(v) for v in vals_by_seed.values()])
    return {"n_seeds": len(m), "seeds": sorted(vals_by_seed), "mean": float(m.mean()), "sd": float(m.std(ddof=1)) if len(m) > 1 else float("nan"),
            "ci95": t_ci(m), "per_seed": {str(k): float(np.mean(v)) for k, v in vals_by_seed.items()},
            "n_cases_per_seed": {str(k): len(v) for k, v in vals_by_seed.items()}}


def paired_compare(a_runs, b_runs, site, split_mode, rng):
    """a_runs/b_runs: {seed: run}. split_mode 'test' (in-federation, seed test split) or 'full' (held-out site).

    Primary inference at the PATIENT level: per-lesion Dice differences averaged over seeds, then
    lesions averaged within patient -> one value per patient; the bootstrap resamples patients and
    the Wilcoxon runs on the patient values. Seed observations of the same lesion and lesions of
    the same patient are never counted as independent."""
    seeds = sorted(set(a_runs) & set(b_runs))
    if not seeds:
        return None
    pairs_by_case, seed_diffs = defaultdict(list), []
    for s in seeds:
        ta, tb = case_table(a_runs[s], site), case_table(b_runs[s], site)
        ka = ta.get(split_mode) or ta.get("test") or ta.get("full")
        kb = tb.get(split_mode) or tb.get("test") or tb.get("full")
        if ka is None or kb is None:
            continue
        if split_mode == "test":   # restrict a 'full' evaluation to this seed's test split
            test_ids = set(a_runs[s]["splits"][site]["test"])
            ka = {c: v for c, v in ka.items() if c in test_ids}; kb = {c: v for c, v in kb.items() if c in test_ids}
        common = sorted(set(ka) & set(kb))
        d = [ka[c]["dice"] - kb[c]["dice"] for c in common]
        for c, v in zip(common, d):
            pairs_by_case[c].append(v)
        if d:
            seed_diffs.append(float(np.mean(d)))
    if not pairs_by_case:
        return None
    n_evals = int(sum(len(v) for v in pairs_by_case.values()))
    lesion_means = {c: float(np.mean(v)) for c, v in pairs_by_case.items()}          # seed-averaged
    by_patient = defaultdict(list)
    for c, v in lesion_means.items():
        by_patient[PATIENT.get(site, {}).get(c, c)].append(v)
    pat = np.array([np.mean(v) for v in by_patient.values()])                        # one value / patient
    try:
        w_p = float(stats.wilcoxon(pat).pvalue) if np.any(pat != 0) else 1.0
    except ValueError:
        w_p = float("nan")
    boots = [pat[rng.randint(len(pat), size=len(pat))].mean() for _ in range(N_BOOT)]
    t_p = float(stats.ttest_rel(seed_diffs, np.zeros(len(seed_diffs))).pvalue) if len(seed_diffs) > 1 else float("nan")
    return {"seeds": seeds, "unit": "patient", "n_patients": int(len(pat)), "n_lesions": len(lesion_means),
            "n_evals": n_evals, "mean_diff": float(pat.mean()),
            "ci95_patient_boot": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wilcoxon_p": w_p, "seed_mean_diffs": seed_diffs, "seed_t_p": t_p,
            "frac_patients_positive": float(np.mean(pat > 0))}


def holm(pvals):
    keys = [k for k, p in pvals.items() if p == p]
    ps = np.array([pvals[k] for k in keys]); order = np.argsort(ps); adj = np.empty(len(ps))
    m = len(ps); prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * ps[i]); prev = max(prev, val); adj[i] = prev
    return {k: float(a) for k, a in zip(keys, adj)}


# ----------------------------------------------------------------------------- main
def main():
    rng = np.random.RandomState(0)
    runs = load_runs()
    if not runs:
        print("no runs"); return
    by = defaultdict(dict)          # (setting, init) -> {seed: run}
    for r in runs:
        by[(setting_of(r), r["init"])][r["seed"]] = r
    print(f"{len(runs)} runs, {len(by)} (setting, init) groups")
    summary = {"n_runs": len(runs), "groups": {}, "paired": {}, "loso": {}, "transfer": {}, "personalized": {}, "convergence": {},
               "comm": {}, "timing": {}, "size_bands": {}, "site_summary": json.load(open(RES / "site_summary.json")) if (RES / "site_summary.json").exists() else {}}

    # ---- per group, per site: seed-level summaries of test-split (in-federation) or full-site (held-out) metrics
    for (setting, init), seeds in sorted(by.items()):
        g = {"setting": setting, "init": init, "seeds": sorted(seeds), "sites": {}}
        for site in SITES:
            for split in ("test", "val", "full"):
                vals = {s: r["results"][site][split] for s, r in seeds.items() if split in r["results"].get(site, {})}
                if not vals:
                    continue
                entry = {}
                for metric in ("dice", "hd95", "ece", "detected", "dice_lcc", "hd95_lcc"):
                    if not all(metric in v for v in vals.values()):
                        continue
                    per = {s: [x for x in v[metric] if x == x] for s, v in vals.items()}
                    if metric.startswith("hd95"):
                        entry[f"{metric}_median_per_seed"] = {str(s): float(np.median(v)) if v else float("nan") for s, v in per.items()}
                        entry[f"{metric}_median_mean"] = float(np.mean([np.median(v) for v in per.values() if v]))
                        entry[f"empty_pred_frac{'_lcc' if metric.endswith('lcc') else ''}"] = float(np.mean([np.mean(np.isnan(v[metric])) for v in vals.values()]))
                    else:
                        entry[metric] = seed_summary(per)
                # volume error, absolute and relative to the reference volume
                av = np.concatenate([np.abs(np.array(v["pred_cm3"]) - np.array(v["gt_cm3"])) for v in vals.values()])
                gv = np.concatenate([np.array(v["gt_cm3"]) for v in vals.values()])
                entry["abs_vol_err_cm3_median"] = float(np.median(av))
                entry["rel_vol_err_median"] = float(np.median(av / np.maximum(gv, 1e-9)))
                # spurious-component burden of the raw prediction
                if all("n_pred_components" in v for v in vals.values()):
                    entry["n_pred_components_median"] = float(np.median(np.concatenate([v["n_pred_components"] for v in vals.values()])))
                # held-out full-site: seed-averaged per-case Dice
                if split == "full":
                    tabs = [case_table(r, site)["full"] for r in seeds.values()]
                    common = sorted(set.intersection(*[set(t) for t in tabs]))
                    entry["dice_case_mean_over_seeds"] = float(np.mean([np.mean([t[c]["dice"] for t in tabs]) for c in common]))
                    # and the seed-specific test-split subset for comparability with in-federation settings
                    sub = {s: [r["results"][site]["full"]["dice"][i] for i, c in enumerate(r["results"][site]["full"]["ids"]) if c in set(r["splits"][site]["test"])]
                           for s, r in seeds.items()}
                    entry["dice_on_seed_test_split"] = seed_summary(sub)
                g["sites"].setdefault(site, {})[split] = entry
        summary["groups"][f"{setting}|{init}"] = g
        # convergence + cost
        hist = {str(s): [(h.get("round", h.get("iter")), h["val_score"]) for h in r["history"] if "val_score" in h] for s, r in seeds.items()}
        summary["convergence"][f"{setting}|{init}"] = hist
        summary["comm"][f"{setting}|{init}"] = {str(s): r.get("comm_bytes", 0) for s, r in seeds.items()}
        summary["timing"][f"{setting}|{init}"] = {str(s): {"train_s": r["time_train_s"], "eval_s": r["time_eval_s"], "total_s": r["time_total_s"]} for s, r in seeds.items()}
        pers = {s: r.get("personalized", {}) for s, r in seeds.items() if r.get("personalized")}
        if pers:
            summary["personalized"][f"{setting}|{init}"] = {site: seed_summary({s: p[site]["test"]["dice"] for s, p in pers.items() if site in p})
                                                            for site in SITES if any(site in p for p in pers.values())}

    # ---- transfer matrix: local model of site A evaluated on site B (full) -- seed-averaged case means
    for init in ("ctfm", "scratch"):
        M = {}
        for a in SITES:
            seeds = by.get((f"local_{a}", init), {})
            if not seeds:
                continue
            for b in SITES:
                if a == b:
                    v = seed_summary({s: r["results"][b]["test"]["dice"] for s, r in seeds.items() if "test" in r["results"].get(b, {})})
                else:
                    v = seed_summary({s: r["results"][b]["full"]["dice"] for s, r in seeds.items() if "full" in r["results"].get(b, {})})
                M[f"{a}->{b}"] = v
        if M:
            summary["transfer"][init] = M

    # ---- paired comparisons
    fam = {}
    for init in ("ctfm", "scratch"):
        for site in SITES:
            L = by.get((f"local_{site}", init), {}); C = by.get((f"central_all", init), {}); Fd = by.get((f"fedavg_all", init), {})
            cl = by.get((f"central_loso_{site}", init), {}); fl = by.get((f"fedavg_loso_{site}", init), {})
            LB = by.get((f"local_{site}_b18000", init), {})
            CB = by.get(("central-balanced_all", init), {})
            comps = {f"fedavg_all-vs-local|{site}|{init}": (Fd, L, "test"),
                     # site-balanced centralized sampling vs volume-proportional pooling
                     f"central_balanced-vs-central_all|{site}|{init}": (CB, C, "test"),
                     f"central_balanced-vs-local|{site}|{init}": (CB, L, "test"),
                     f"fedavg_all-vs-central_all|{site}|{init}": (Fd, C, "test"),
                     f"central_all-vs-local|{site}|{init}": (C, L, "test"),
                     f"fedavg_loso-vs-central_loso|{site}|{init}": (fl, cl, "full"),
                     f"central_all-vs-central_loso|{site}|{init}": (C, cl, "test"),
                     f"fedavg_all-vs-fedavg_loso|{site}|{init}": (Fd, fl, "test"),
                     # held-out models against the site's own local model, on the identical test cases
                     f"central_loso-vs-local|{site}|{init}": (cl, L, "test"),
                     f"fedavg_loso-vs-local|{site}|{init}": (fl, L, "test"),
                     # budget sensitivity: 18k-iteration local training vs the 6k default
                     f"local_b18000-vs-local|{site}|{init}": (LB, L, "test")}
            for a in SITES:
                if a != site:
                    comps[f"central_loso-vs-local_{a}|{site}|{init}"] = (cl, by.get((f"local_{a}", init), {}), "full")
            for key, (A, B, mode) in comps.items():
                if A and B:
                    res = paired_compare(A, B, site, mode, rng)
                    if res:
                        summary["paired"][key] = res; fam.setdefault(key.split("|")[0], {})[key] = res["wilcoxon_p"]
        for site in SITES:   # init effect: key = comparison|site|init with init = "both"
            for setting, mode in ((f"local_{site}", "test"), ("central_all", "test"), ("fedavg_all", "test"),
                                  (f"central_loso_{site}", "full"), (f"fedavg_loso_{site}", "full"),
                                  (f"local_{site}_b18000", "test")):
                A, B = by.get((setting, "ctfm"), {}), by.get((setting, "scratch"), {})
                if A and B:
                    res = paired_compare(A, B, site, mode, rng)
                    if res:
                        short = setting.replace(f"_{site}", "")
                        key = f"ctfm-vs-scratch_{short}|{site}|both"; summary["paired"][key] = res; fam.setdefault("ctfm-vs-scratch", {})[key] = res["wilcoxon_p"]
    # ---- ablations and personalization vs FedAvg (ctfm init), test splits
    Fd = by.get(("fedavg_all", "ctfm"), {})
    if Fd:
        pers = {}
        for s_, r in Fd.items():
            if r.get("personalized"):
                pers[s_] = {"results": {site: {"test": v["test"]} for site, v in r["personalized"].items()}, "splits": r["splits"]}
        arms = {"fedavg_ft-vs-fedavg_all": pers, "fedprox-vs-fedavg_all": by.get(("fedprox0.01_all", "ctfm"), {}),
                "fedavg_uniform-vs-fedavg_all": by.get(("fedavg-uniform_all", "ctfm"), {}), "fedavg_r60e100-vs-fedavg_all": by.get(("fedavg-r60e100_all", "ctfm"), {}),
                "fedbn-vs-fedavg_all": by.get(("fedbn_all", "ctfm"), {}),
                "fedavg_optreset-vs-fedavg_all": by.get(("fedavg-optreset_all", "ctfm"), {})}
        Fb = by.get(("fedbn_all", "ctfm"), {})
        if Fb and any(r.get("personalized") for r in Fb.values()):
            arms["fedbn_ft-vs-fedavg_all"] = {s_: {"results": {site: {"test": v["test"]} for site, v in r["personalized"].items()}, "splits": r["splits"]}
                                              for s_, r in Fb.items() if r.get("personalized")}
        for name, A in arms.items():
            if not A:
                continue
            for site in SITES:
                res = paired_compare(A, Fd, site, "test", rng)
                if res:
                    key = f"{name}|{site}|ctfm"; summary["paired"][key] = res; fam.setdefault("ablation", {})[key] = res["wilcoxon_p"]
    for family, pv in fam.items():
        for k, adj in holm(pv).items():
            summary["paired"][k]["wilcoxon_p_holm"] = adj

    # ---- Dice by tumour-volume band, in-federation test splits. Unit = unique lesion (Dice averaged
    # over the seeds whose test split contains it); n_lesions / n_patients / n_evals all reported.
    bands = [("<1", 0, 1), ("1-10", 1, 10), ("10-50", 10, 50), (">=50", 50, 1e9)]
    for (setting, init), seeds in by.items():
        for site in SITES:
            acc, vol = defaultdict(list), {}
            for r in seeds.values():
                res = r["results"].get(site, {}).get("test")
                if not res:
                    continue
                for i, cid in enumerate(res["ids"]):
                    acc[cid].append(res["dice"][i]); vol[cid] = res["gt_cm3"][i]
            if not acc:
                continue
            out = {}
            for b, lo, hi in bands:
                ids = [c for c in acc if lo <= vol[c] < hi]
                out[b] = {"n_lesions": len(ids),
                          "n_patients": len({PATIENT.get(site, {}).get(c, c) for c in ids}),
                          "n_evals": int(sum(len(acc[c]) for c in ids)),
                          "dice": float(np.mean([np.mean(acc[c]) for c in ids])) if ids else None}
            summary["size_bands"][f"{setting}|{init}|{site}"] = out

    json.dump(summary, open(RES / "summary_fl.json", "w"), indent=1)
    write_report(summary)
    write_tables(summary)
    print("wrote", RES / "summary_fl.json")


# ----------------------------------------------------------------------------- outputs
def fmt(e, key="dice"):
    if not e or key not in e:
        return "--"
    s = e[key]
    return f"{s['mean']:.3f} $\\pm$ {s['sd']:.3f}" if s["n_seeds"] > 1 else f"{s['mean']:.3f} (n=1)"


def write_report(S):
    lines = ["# Study 3 aggregate report", f"runs: {S['n_runs']}", ""]
    for init in ("ctfm", "scratch"):
        lines.append(f"## init = {init}\n")
        lines.append("| setting | site | split | Dice mean+/-SD (seeds) | HD95 med | empty | ECE | detect |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, g in S["groups"].items():
            if g["init"] != init:
                continue
            for site, sp in g["sites"].items():
                for split, e in sp.items():
                    if split == "val":
                        continue
                    d = e["dice"]
                    lines.append(f"| {g['setting']} | {site} | {split} | {d['mean']:.3f} +/- {d['sd']:.3f} (n={d['n_seeds']}) | {e['hd95_median_mean']:.1f} | {e['empty_pred_frac']:.2f} | {e['ece']['mean']:.3f} | {e['detected']['mean']:.2f} |")
        lines.append("")
    lines.append("## Paired comparisons (Dice, A minus B; patient-level inference)\n")
    lines.append("| comparison | site | init | n pat / les / evals | mean diff | 95% CI (patient boot) | Wilcoxon p | Holm p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, p in S["paired"].items():
        c, site, init = k.split("|")
        lines.append(f"| {c} | {site} | {init} | {p['n_patients']}/{p['n_lesions']}/{p['n_evals']} | {p['mean_diff']:+.3f} | [{p['ci95_patient_boot'][0]:+.3f}, {p['ci95_patient_boot'][1]:+.3f}] | {p['wilcoxon_p']:.2g} | {p.get('wilcoxon_p_holm', float('nan')):.2g} |")
    if S["transfer"]:
        lines.append("\n## Transfer matrix (local model rows -> evaluated site columns; diagonal = own test split, off-diagonal = full site)\n")
        for init, M in S["transfer"].items():
            lines.append(f"init {init}:")
            for a in SITES:
                lines.append("  " + a + ": " + "  ".join(f"{b}={M[f'{a}->{b}']['mean']:.3f}" for b in SITES if f"{a}->{b}" in M))
    if S["personalized"]:
        lines.append("\n## Personalized (FedAvg + local fine-tuning) test Dice\n")
        for k, v in S["personalized"].items():
            lines.append(f"{k}: " + "  ".join(f"{site}={e['mean']:.3f}+/-{e['sd']:.3f}" for site, e in v.items()))
    open(RES / "REPORT.md", "w").write("\n".join(lines) + "\n")


def write_tables(S):
    TAB.mkdir(parents=True, exist_ok=True)
    G = S["groups"]
    def cell(setting, init, site, split="test", key="dice"):
        g = G.get(f"{setting}|{init}")
        if not g or site not in g["sites"] or split not in g["sites"][site]:
            return "--"
        return fmt(g["sites"][site][split], key)
    rows = [("Local (own site)", lambda s: f"local_{s}", "test"), ("Centralized, all sites", lambda s: "central_all", "test"),
            ("FedAvg, all sites", lambda s: "fedavg_all", "test"),
            ("Centralized, site held out", lambda s: f"central_loso_{s}", "full"), ("FedAvg, site held out", lambda s: f"fedavg_loso_{s}", "full")]
    out = ["\\begin{table}[t]\\centering\\caption{Test Dice (mean $\\pm$ SD over three seeds) per site and training setting. Held-out rows evaluate the whole site.}",
           "\\label{tab:main}\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llccc}\\toprule", "Init & Setting & TW & NL & US \\\\ \\midrule"]
    for init in ("ctfm", "scratch"):
        for name, f, split in rows:
            out.append(f"{'CT-FM' if init == 'ctfm' else 'Scratch'} & {name} & " + " & ".join(cell(f(s), init, s, split) for s in SITES) + " \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule\\end{tabular}\\end{table}"
    open(TAB / "tab_main.tex", "w").write("\n".join(out) + "\n")
    # paired table (full listing -> supplement; longtable so it never overflows a page)
    out = ["% Full paired-comparison listing. Belongs in the SUPPLEMENT; the main text shows a forest plot of the primary comparisons.",
           "\\begin{longtable}{llcccc}",
           "\\caption{Paired Dice differences (A $-$ B), patient-level inference: lesion differences averaged over seeds, lesions averaged within patient; 95\\% bootstrap CIs resample patients and Wilcoxon signed-rank tests run on the patient values, Holm-adjusted within each comparison family.}\\label{tab:paired}\\\\",
           "\\toprule Comparison & Site & Init & $n$ (patients) & Mean diff [95\\% CI] & $p$ (Holm) \\\\ \\midrule \\endfirsthead",
           "\\toprule Comparison & Site & Init & $n$ (patients) & Mean diff [95\\% CI] & $p$ (Holm) \\\\ \\midrule \\endhead",
           "\\bottomrule \\endlastfoot"]
    for k, p in S["paired"].items():
        c, site, init = k.split("|")
        out.append(f"{c.replace('_', ' ').replace('-vs-', ' vs ')} & {site} & {init} & {p['n_patients']} & {p['mean_diff']:+.3f} [{p['ci95_patient_boot'][0]:+.3f}, {p['ci95_patient_boot'][1]:+.3f}] & {p.get('wilcoxon_p_holm', float('nan')):.3g} \\\\")
    out.append("\\end{longtable}")
    open(TAB / "tab_paired.tex", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
