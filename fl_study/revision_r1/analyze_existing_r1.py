#!/usr/bin/env python3
"""Revision-1 analyses that need no new training or inference: they read the run JSONs of the
submitted grid (results/fl) only.

  1. Communication to a SHARED validation score (Reviewer 3, minor 4). For FedAvg on all three
     sites, CT-FM versus random initialization, per seed: the first evaluated round at which the
     mean validation Dice reaches a common target, for fixed targets and for the target defined
     by the weaker arm's own best score in that seed. Validation was evaluated every second
     round, so the resolution is two rounds; "not reached" is reported as such.
  2. Initialization effect on US cases by pre-training exposure (Reviewers 2-4). 9 of the 144 US
     CT series are absent from the published CT-FM scan list. For models that never trained on
     US (two-site centralized and federated models with US held out, and the single-site IH and
     NL models), all 144 US cases are evaluated in every seed, so the paired CT-FM minus random
     Dice difference can be stratified by exposure at the patient level. For models trained on
     US the same stratification uses the seed test splits. The unexposed stratum is small and
     differs in slice spacing, so this is descriptive.

Writes results/fl_r1/existing_r1.json.
"""
from __future__ import annotations
import csv, json, pickletools, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "fl_study"))
import aggregate_fl as A  # noqa: E402

OUT = ROOT / "results" / "fl_r1"
N_BOOT = 2000


def comm_to_shared_target(by):
    out = {"targets": {}, "per_seed": {}, "note": "validation evaluated every 2 rounds; one round moves 2*K*|theta| values"}
    C, S = by.get(("fedavg_all", "ctfm"), {}), by.get(("fedavg_all", "scratch"), {})
    fixed = [0.40, 0.45, 0.50, 0.55]

    def first_round(r, target):
        for h in r["history"]:
            if "val_score" in h and h["val_score"] >= target:
                return int(h["round"])
        return None

    for seed in sorted(set(C) & set(S)):
        rc, rs = C[seed], S[seed]
        per_round = rc["comm_bytes"] / rc["config"]["rounds"]
        best_c = max(h["val_score"] for h in rc["history"] if "val_score" in h)
        best_s = max(h["val_score"] for h in rs["history"] if "val_score" in h)
        rec = {"best_val_ctfm": best_c, "best_val_random": best_s, "gb_per_round": per_round / 1e9, "targets": {}}
        for name, t in [("random_best", min(best_c, best_s))] + [(f"{x:.2f}", x) for x in fixed]:
            a, b = first_round(rc, t), first_round(rs, t)
            rec["targets"][name] = {"target": t, "round_ctfm": a, "round_random": b,
                                    "gb_ctfm": None if a is None else a * per_round / 1e9,
                                    "gb_random": None if b is None else b * per_round / 1e9}
        out["per_seed"][str(seed)] = rec
    for name in out["per_seed"][next(iter(out["per_seed"]))]["targets"] if out["per_seed"] else []:
        rc = [v["targets"][name]["round_ctfm"] for v in out["per_seed"].values()]
        rs = [v["targets"][name]["round_random"] for v in out["per_seed"].values()]
        out["targets"][name] = {"rounds_ctfm": rc, "rounds_random": rs,
                                "n_reached_ctfm": sum(x is not None for x in rc), "n_reached_random": sum(x is not None for x in rs),
                                "mean_round_ctfm": float(np.mean([x for x in rc if x is not None])) if any(x is not None for x in rc) else None,
                                "mean_round_random": float(np.mean([x for x in rs if x is not None])) if any(x is not None for x in rs) else None}
    return out


def us_exposure():
    """case id -> True when the CT series is in the CT-FM scan list."""
    D = ROOT / "data" / "external" / "ctfm_pretraining"
    strings = []
    with open(D / "ctfm_scan_list.pkl", "rb") as f:          # never unpickled: constants read by the disassembler
        for op, arg, pos in pickletools.genops(f):
            if isinstance(arg, str):
                strings.append(arg)
    uids = {s for s in strings if s.startswith("1.") and re.fullmatch(r"\d+(\.\d+)+", s)}
    audit = json.load(open(ROOT / "data" / "external" / "nsclc_radiogenomics" / "seg_audit.json"))
    rg = list(csv.DictReader(open(ROOT / "data" / "external_radiogenomics_processed_1mm" / "manifest_rg.csv")))
    return uids, audit, rg


def strat_effect(by, exposed, rng):
    """Paired CT-FM minus random Dice on US, by exposure stratum, patient level (US is 1 mask per patient)."""
    res = {}
    settings = [("central_loso_US", "full"), ("fedavg_loso_US", "full"), ("local_TW", "full"), ("local_NL", "full"),
                ("local_US", "test"), ("central_all", "test"), ("fedavg_all", "test")]
    for setting, split in settings:
        C, S = by.get((setting, "ctfm"), {}), by.get((setting, "scratch"), {})
        d = defaultdict(list)
        for seed in sorted(set(C) & set(S)):
            ta, tb = A.case_table(C[seed], "US").get(split), A.case_table(S[seed], "US").get(split)
            if not ta or not tb:
                continue
            for c in set(ta) & set(tb):
                d[c].append(ta[c]["dice"] - tb[c]["dice"])
        entry = {"split": split}
        for name, flag in (("exposed", True), ("unexposed", False)):
            v = np.array([np.mean(x) for c, x in d.items() if exposed.get(c) is flag])
            if len(v) == 0:
                entry[name] = {"n_patients": 0}; continue
            boots = [v[rng.randint(len(v), size=len(v))].mean() for _ in range(N_BOOT)]
            entry[name] = {"n_patients": int(len(v)), "n_evals": int(sum(len(x) for c, x in d.items() if exposed.get(c) is flag)),
                           "mean_diff": float(v.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        res[setting] = entry
    return res


def main():
    rng = np.random.RandomState(20261116)
    runs = A.load_runs()
    by = defaultdict(dict)
    for r in runs:
        by[(A.setting_of(r), r["init"])][r["seed"]] = r
    out = {"communication": comm_to_shared_target(by)}

    uids, audit, rg = us_exposure()
    man = {r["case"]: r["id"] for r in json.load(open(ROOT / "data" / "fl_cache" / "US" / "manifest.json"))}
    exposed = {man[a["pid"]]: a["ref"] in uids for a in audit}          # cache case id -> series in the CT-FM scan list
    n_unexp = sum(1 for v in exposed.values() if not v)
    print(f"US cases mapped: {len(exposed)}/{len(man)}; unexposed: {n_unexp}")
    assert len(exposed) == len(man) == 144 and n_unexp == 9, "exposure mapping must reproduce 135/144"
    thick = {man[r["pid"]]: float(r["slice_thickness"]) if r["slice_thickness"] else float("nan") for r in rg}
    out["us_exposure"] = {"n_exposed": len(exposed) - n_unexp, "n_unexposed": n_unexp,
                          "unexposed_slice_thickness": sorted(thick.get(c, float("nan")) for c, v in exposed.items() if not v),
                          "exposed_slice_thickness_min": float(np.nanmin([thick.get(c, np.nan) for c, v in exposed.items() if v])) if thick else None,
                          "effect": strat_effect(by, exposed, rng)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "existing_r1.json", "w"), indent=1)
    print(json.dumps(out["communication"]["targets"], indent=1))
    print(json.dumps(out["us_exposure"], indent=1))


if __name__ == "__main__":
    main()
