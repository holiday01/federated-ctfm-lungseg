#!/usr/bin/env python3
"""Revision-1 aggregation of the NEW training arms -> results/fl_r1/train_summary.json (+ .md digest).

  A  extended budget: centralized pooling and FedAvg at 18,000 iterations per site against the
     6,000-iteration runs; initialization effect at both budgets and its change (difference in
     differences); the same for the budget-matched local runs of the first submission.
  B  fixed-split replications: split of seed 7, five training runs per configuration (run seeds
     7 / 11 / 23 / 101 / 211); run-to-run SD on the fixed split beside the SD over the three seeds
     that also redraw the split; the main paired comparisons on the fixed test set.
  F  FedBN on two sites, third held out, under the normalization rules of eval_fedbn_heldout.py.
  G  FedProx with mu = 0.001 / 0.01 / 0.1.
  H' communication to a shared validation score in the 90-round federations.

Rules that keep the first submission intact:
  * comparisons that already exist in results/fl/summary_fl.json are COPIED from it, never recomputed;
  * every new comparison uses this script's own bootstrap RNG and its own Holm family.
Sections are skipped while their runs do not exist; incomplete groups are listed under "incomplete".
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
FIX = ROOT / "results" / "fl_fixsplit"
OUT = R1 / "train_summary.json"
SITES, SEEDS = A.SITES, A.SEEDS
FIX_SPLIT_SEED, FIX_RUN_SEEDS = 7, [7, 11, 23, 101, 211]
MUS = ["0.001", "0.01", "0.1"]


def dice_tab(runs, site, split="test"):
    """{seed: {case: dice}}; split 'test' restricts a full-site evaluation to the seed's test split."""
    out = {}
    for s, r in runs.items():
        t = A.case_table(r, site)
        k = t.get(split) or (t.get("full") if split == "test" else None)
        if k is None:
            continue
        if split == "test" and "test" not in t:
            ids = set(r["splits"][site]["test"]); k = {c: v for c, v in k.items() if c in ids}
        out[s] = {c: v["dice"] for c, v in k.items()}
    return out


def minus(TA, TB):
    return {s: {c: TA[s][c] - TB[s][c] for c in set(TA[s]) & set(TB[s])} for s in set(TA) & set(TB)}


def old(S0, key):
    """A comparison of the first submission, copied verbatim."""
    p = S0["paired"].get(key)
    if not p:
        return None
    return {"source": "summary_fl.json (first submission)", "n_patients": p["n_patients"], "mean_diff": p["mean_diff"], "ci95": p["ci95_patient_boot"],
            "wilcoxon_p": p["wilcoxon_p"], "wilcoxon_p_holm": p.get("wilcoxon_p_holm"), "seed_mean_diffs": p["seed_mean_diffs"]}


def first_round(r, target):
    for h in r["history"]:
        if "val_score" in h and h["val_score"] >= target:
            return int(h["round"])
    return None


def main():
    rng = np.random.RandomState(20261117)
    S0 = json.load(open(A.RES / "summary_fl.json"))
    by = defaultdict(dict)
    for r in A.load_runs():
        by[(A.setting_of(r), r["init"])][r["seed"]] = r
    S = {"incomplete": {}, "budget": {}, "budget_paired": {}, "comm90": {}, "fixsplit": {}, "fixsplit_paired": {}, "fedbn": {}, "fedbn_paired": {},
         "fedprox": {}, "fedprox_paired": {}}
    fam = defaultdict(dict)

    def need(setting, init, n=3):
        g = by.get((setting, init), {})
        if 0 < len(g) < n:
            S["incomplete"][f"{setting}|{init}"] = sorted(g)
        return g

    def add(section, family, key, res):
        if res:
            S[section][key] = res
            if family:
                fam[family][key] = res

    # ------------------------------------------------------------------ A: extended budget
    arms = {"local": (lambda s: f"local_{s}", lambda s: f"local_{s}_b18000"),
            "central": (lambda s: "central_all", lambda s: "central_all_b18000"),
            "fedavg": (lambda s: "fedavg_all", lambda s: "fedavg-r90e200_all_b18000")}
    for arm, (f6, f18) in arms.items():
        for site in SITES:
            T = {}
            for init in ("ctfm", "scratch"):
                g6, g18 = need(f6(site), init), need(f18(site), init)
                T[init] = (dice_tab(g6, site), dice_tab(g18, site))
                for name, t in (("6k", T[init][0]), ("18k", T[init][1])):
                    if t:
                        S["budget"][f"{arm}|{init}|{site}|{name}"] = seed_stat(t)
                if arm == "local":
                    S["budget_paired"][f"gain18k-6k|{arm}|{init}|{site}"] = old(S0, f"local_b18000-vs-local|{site}|{init}")
                elif T[init][0] and T[init][1]:
                    add("budget_paired", "r1_budget_gain", f"gain18k-6k|{arm}|{init}|{site}", paired(T[init][1], T[init][0], site, rng))
            short = {"local": "local", "central": "central_all", "fedavg": "fedavg_all"}[arm]
            S["budget_paired"][f"init6k|{arm}|{site}"] = old(S0, f"ctfm-vs-scratch_{short}|{site}|both")
            if arm == "local":
                S["budget_paired"][f"init18k|{arm}|{site}"] = old(S0, f"ctfm-vs-scratch_local_b18000|{site}|both")
            elif T["ctfm"][1] and T["scratch"][1]:
                add("budget_paired", "r1_budget_init18k", f"init18k|{arm}|{site}", paired(T["ctfm"][1], T["scratch"][1], site, rng))
            if all(T[i][j] for i in T for j in (0, 1)):
                # change of the initialization effect with budget: (CT-FM - random at 18k) - (CT-FM - random at 6k)
                add("budget_paired", "r1_budget_did", f"did|{arm}|{site}",
                    paired(minus(T["ctfm"][1], T["scratch"][1]), minus(T["ctfm"][0], T["scratch"][0]), site, rng))
    for init in ("ctfm", "scratch"):
        for site in SITES:
            C, Fd, L = (dice_tab(by.get((k, init), {}), site) for k in ("central_all_b18000", "fedavg-r90e200_all_b18000", f"local_{site}_b18000"))
            if C and Fd:
                add("budget_paired", "r1_budget_fed_vs_central18k", f"fedavg-vs-central18k|{init}|{site}", paired(Fd, C, site, rng))
            for name, X in (("central", C), ("fedavg", Fd)):
                if X and L:
                    add("budget_paired", "r1_budget_vs_local18k", f"{name}-vs-local18k|{init}|{site}", paired(X, L, site, rng))

    # ------------------------------------------------------------------ H': shared validation target in the 90-round federations
    C90, S90 = by.get(("fedavg-r90e200_all_b18000", "ctfm"), {}), by.get(("fedavg-r90e200_all_b18000", "scratch"), {})
    for seed in sorted(set(C90) & set(S90)):
        rc, rs = C90[seed], S90[seed]
        per_round = rc["comm_bytes"] / rc["config"]["rounds"]
        bc = max(h["val_score"] for h in rc["history"] if "val_score" in h); bs = max(h["val_score"] for h in rs["history"] if "val_score" in h)
        rec = {"best_val_ctfm": bc, "best_val_random": bs, "best_round_ctfm": rc["best"].get("round"), "best_round_random": rs["best"].get("round"),
               "gb_per_round": per_round / 1e9, "eval_every_rounds": rc["config"]["eval_every_rounds"], "targets": {}}
        for name, t in [("random_best", min(bc, bs))] + [(f"{x:.2f}", x) for x in (0.40, 0.45, 0.50, 0.55, 0.60)]:
            a, b = first_round(rc, t), first_round(rs, t)
            rec["targets"][name] = {"target": t, "round_ctfm": a, "round_random": b}
        S["comm90"][str(seed)] = rec

    # ------------------------------------------------------------------ B: fixed-split replications
    fx = defaultdict(dict)
    for f in sorted(glob.glob(str(FIX / "*.json"))):
        r = json.load(open(f))
        if "results" not in r:
            continue
        assert r.get("split_seed", r["config"].get("split_seed")) == FIX_SPLIT_SEED, f
        fx[(A.setting_of(r), r["init"])][r["seed"]] = r
    for (setting, init), g in list(fx.items()):          # the existing seed-7 run is the fifth replicate (split seed = run seed = 7)
        base = by.get((setting, init), {}).get(FIX_SPLIT_SEED)
        if base is not None:
            g[FIX_SPLIT_SEED] = base
        if len(g) < len(FIX_RUN_SEEDS):
            S["incomplete"][f"fixsplit:{setting}|{init}"] = sorted(g)
    for (setting, init), g in sorted(fx.items()):
        for site in SITES:
            t = dice_tab(g, site)
            t = {s: v for s, v in t.items() if "test" in g[s]["results"].get(site, {})}
            if not t:
                continue
            ids = [frozenset(v) for v in t.values()]
            assert all(i == ids[0] for i in ids), f"fixed split differs between replicates: {setting} {init} {site}"
            e = seed_stat(t)
            o = S0["groups"].get(f"{setting}|{init}", {}).get("sites", {}).get(site, {}).get("test", {}).get("dice")
            S["fixsplit"][f"{setting}|{init}|{site}"] = {"run_seeds": sorted(t), "n_cases": len(ids[0]), "mean": e["mean"], "sd_runs_fixed_split": e["sd"],
                                                         "per_run": e["per_seed"], "min": float(min(e["per_seed"].values())), "max": float(max(e["per_seed"].values())),
                                                         "mean_3seeds_resplit": o["mean"] if o else None, "sd_3seeds_resplit": o["sd"] if o else None}
    comps = {}
    for init in ("ctfm", "scratch"):
        for site in SITES:
            comps[f"fedavg_all-vs-local|{site}|{init}"] = (("fedavg_all", init), (f"local_{site}", init), f"fedavg_all-vs-local|{site}|{init}")
            comps[f"central_all-vs-local|{site}|{init}"] = (("central_all", init), (f"local_{site}", init), f"central_all-vs-local|{site}|{init}")
            comps[f"fedavg_all-vs-central_all|{site}|{init}"] = (("fedavg_all", init), ("central_all", init), f"fedavg_all-vs-central_all|{site}|{init}")
    for site in SITES:
        for short, setting in (("local", f"local_{site}"), ("central_all", "central_all"), ("fedavg_all", "fedavg_all")):
            comps[f"ctfm-vs-scratch_{short}|{site}|both"] = ((setting, "ctfm"), (setting, "scratch"), f"ctfm-vs-scratch_{short}|{site}|both")
    for key, (ka, kb, oldkey) in comps.items():
        site = key.split("|")[1]
        TA, TB = dice_tab(fx.get(ka, {}), site), dice_tab(fx.get(kb, {}), site)
        if not TA or not TB:
            continue
        res = paired(TA, TB, site, rng)
        if not res:
            continue
        d = np.array(res["seed_mean_diffs"])
        res.update({"run_seeds": sorted(set(TA) & set(TB)), "run_diff_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                    "run_diff_min": float(d.min()), "run_diff_max": float(d.max()),
                    "n_runs_same_sign_as_mean": int(np.sum(np.sign(d) == np.sign(res["mean_diff"]))), "first_submission": old(S0, oldkey)})
        add("fixsplit_paired", "r1_fixsplit_" + key.split("|")[0], key, res)

    # ------------------------------------------------------------------ F: FedBN held-out rules
    fb = defaultdict(dict)
    for f in sorted(glob.glob(str(R1 / "fedbn_heldout" / "*.json"))):
        r = json.load(open(f)); fb[r["heldout"]][r["seed"]] = r
    for held, g in sorted(fb.items()):
        if len(g) < 3:
            S["incomplete"][f"fedbn_heldout:{held}"] = sorted(g)
        names = list(next(iter(g.values()))["rules"])
        T = {n: {s: dict(zip(r["rules"][n]["ids"], r["rules"][n]["dice"])) for s, r in g.items()} for n in names}
        ent = {"seeds": sorted(g), "train_sites": next(iter(g.values()))["train_sites"], "rules": {n: seed_stat(T[n]) for n in names},
               "verify_max_abs_dice_diff": float(max(r.get("verify", {}).get("max_abs_dice_diff", float("nan")) for r in g.values()))}
        means = {n: ent["rules"][n]["mean"] for n in names}
        ent["range_across_rules"] = [float(min(means.values())), float(max(means.values()))]
        ent["best_rule"], ent["worst_rule"] = max(means, key=means.get), min(means, key=means.get)
        for comp in ("fedavg_loso", "central_loso"):
            c = {s: dict(zip(r["results"][held]["full"]["ids"], r["results"][held]["full"]["dice"])) for s, r in by.get((f"{comp}_{held}", "ctfm"), {}).items()}
            if c:
                ent[comp] = seed_stat(c)
                if comp == "fedavg_loso":
                    for n in names:
                        add("fedbn_paired", "r1_fedbn_vs_fedavg_loso", f"{n}-vs-fedavg_loso|{held}", paired(T[n], c, held, rng))
        for n in names:
            if n != "avg_size":
                add("fedbn_paired", "r1_fedbn_rule_vs_avg_size", f"{n}-vs-avg_size|{held}", paired(T[n], T["avg_size"], held, rng))
        S["fedbn"][held] = ent

    # ------------------------------------------------------------------ G: FedProx grid
    Fd = by.get(("fedavg_all", "ctfm"), {})
    for mu in MUS:
        g = need(f"fedprox{mu}_all", "ctfm")
        if not g:
            continue
        for site in SITES:
            t = dice_tab(g, site)
            S["fedprox"][f"{mu}|{site}"] = dict(seed_stat(t), best_rounds={str(s): r["best"].get("round") for s, r in g.items()})
            if mu == "0.01":
                S["fedprox_paired"][f"fedprox{mu}-vs-fedavg_all|{site}"] = old(S0, f"fedprox-vs-fedavg_all|{site}|ctfm")
            else:
                add("fedprox_paired", "r1_fedprox", f"fedprox{mu}-vs-fedavg_all|{site}", paired(t, dice_tab(Fd, site), site, rng))
    for site in SITES:
        S["fedprox"][f"fedavg|{site}"] = seed_stat(dice_tab(Fd, site))

    for family, d in fam.items():
        adj = A.holm({k: v["wilcoxon_p"] for k, v in d.items()})
        for k, a in adj.items():
            d[k]["wilcoxon_p_holm"] = a; d[k]["holm_family"] = family; d[k]["holm_family_size"] = len(adj)
    json.dump(S, open(OUT, "w"), indent=1)
    print("wrote", OUT)

    # ------------------------------------------------------------------ digest for human review
    def pd(p):
        if not p:
            return "--"
        h = p.get("wilcoxon_p_holm")
        return f"{p['mean_diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] p_holm={h:.2g}" if h is not None else f"{p['mean_diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"

    def ms(e):
        return "--" if not e else (f"{e['mean']:.3f} +/- {e['sd']:.3f} (n={e['n_seeds']})" if e["n_seeds"] > 1 else f"{e['mean']:.3f} (n=1)")

    L = ["# Revision 1: new training arms (A, B, F, G, H')", "", "incomplete groups: " + (json.dumps(S["incomplete"]) if S["incomplete"] else "none"), "",
         "## A. Extended budget (test Dice, mean +/- SD over seeds)", "", "| arm | site | CT-FM 6k | CT-FM 18k | random 6k | random 18k | init effect 6k | init effect 18k | change (DiD) |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        for site in SITES:
            B, P = S["budget"], S["budget_paired"]
            L.append(f"| {arm} | {site} | " + " | ".join(ms(B.get(f"{arm}|{i}|{site}|{b}")) for i in ("ctfm", "scratch") for b in ("6k", "18k"))
                     + f" | {pd(P.get(f'init6k|{arm}|{site}'))} | {pd(P.get(f'init18k|{arm}|{site}'))} | {pd(P.get(f'did|{arm}|{site}'))} |")
    L += ["", "| comparison | " + " | ".join(SITES) + " |", "|---|---|---|---|"]
    for name in sorted({k.rsplit("|", 1)[0] for k in S["budget_paired"] if k.split("|")[0] not in ("init6k", "init18k", "did")}):
        L.append(f"| {name} | " + " | ".join(pd(S["budget_paired"].get(f"{name}|{s}")) for s in SITES) + " |")
    if S["comm90"]:
        L += ["", "## H'. 90-round federations: first evaluated round reaching the target (CT-FM / random)", ""]
        for seed, v in S["comm90"].items():
            L.append(f"seed {seed}: best val CT-FM {v['best_val_ctfm']:.3f} (round {v['best_round_ctfm']}), random {v['best_val_random']:.3f} (round {v['best_round_random']}); "
                     + "; ".join(f"{n}: {t['round_ctfm']} / {t['round_random']}" for n, t in v["targets"].items()))
    if S["fixsplit"]:
        L += ["", "## B. Fixed split (seed 7), five training runs", "", "| setting | init | site | runs | mean | SD over runs (fixed split) | min-max | SD over 3 seeds (split redrawn) |", "|---|---|---|---|---|---|---|---|"]
        for k, e in S["fixsplit"].items():
            st, init, site = k.split("|")
            sd3 = "--" if e["sd_3seeds_resplit"] is None else f"{e['sd_3seeds_resplit']:.3f}"
            L.append(f"| {st} | {init} | {site} | {len(e['run_seeds'])} | {e['mean']:.3f} | {e['sd_runs_fixed_split']:.3f} | {e['min']:.3f}-{e['max']:.3f} | {sd3} |")
        L += ["", "| comparison | site | init | fixed split: diff [CI] | per-run diffs min..max | runs with the sign of the mean | first submission |", "|---|---|---|---|---|---|---|"]
        for k, p in S["fixsplit_paired"].items():
            c, site, init = k.split("|")
            L.append(f"| {c} | {site} | {init} | {pd(p)} | {p['run_diff_min']:+.3f}..{p['run_diff_max']:+.3f} | {p['n_runs_same_sign_as_mean']}/{len(p['run_seeds'])} | {pd(p['first_submission'])} |")
    if S["fedbn"]:
        L += ["", "## F. FedBN, held-out site, by normalization rule (full-site Dice)", ""]
        for held, e in S["fedbn"].items():
            L.append(f"held out {held} (trained on {'-'.join(e['train_sites'])}, seeds {e['seeds']}): " + "; ".join(f"{n} {ms(v)}" for n, v in e["rules"].items())
                     + f"; FedAvg two-site {ms(e.get('fedavg_loso'))}; centralized two-site {ms(e.get('central_loso'))}; range {e['range_across_rules'][0]:.3f}-{e['range_across_rules'][1]:.3f}")
        for k, p in S["fedbn_paired"].items():
            L.append(f"- {k}: {pd(p)}")
    if any(k.split("|")[0] in MUS for k in S["fedprox"]):
        L += ["", "## G. FedProx (CT-FM, test Dice)", "", "| mu | " + " | ".join(SITES) + " | vs FedAvg: " + " / ".join(SITES) + " |", "|---|---|---|---|---|"]
        L.append("| FedAvg | " + " | ".join(ms(S["fedprox"].get(f"fedavg|{s}")) for s in SITES) + " | |")
        for mu in MUS:
            if f"{mu}|TW" in S["fedprox"]:
                L.append(f"| {mu} | " + " | ".join(ms(S["fedprox"].get(f"{mu}|{s}")) for s in SITES) + " | "
                         + " / ".join(pd(S["fedprox_paired"].get(f"fedprox{mu}-vs-fedavg_all|{s}")) for s in SITES) + " |")
    open(OUT.with_suffix(".md"), "w").write("\n".join(L) + "\n")
    print("wrote", OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
