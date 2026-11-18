#!/usr/bin/env python3
"""FedBN trained on two sites with the third held out: the held-out site under several
normalization rules (Reviewer 3, minor 3). Inference only, from the checkpoints of
jobs_r1_fedbn_loso.json (data/fl_ckpt/fedbn_<A>-<B>_ctfm_s<seed>.pt).

FedBN keeps every BatchNorm tensor (affine weight and bias, running mean and variance) local to
its client, so a site that never trained has no normalization state of its own. All rules use
the same averaged non-BN tensors of the selected round and differ only in the BN tensors:

  avg_size      cohort-size-weighted average of the two clients' BN tensors. This is the rule of
                train_fl.py; the recomputed Dice is checked against the value stored in the run JSON.
  avg_uniform   unweighted average of the two clients' BN tensors.
  client_<A>, client_<B>   the BN tensors of one client.
  target_stats  affine parameters as in avg_size; running mean and variance re-estimated on the
                UNLABELED images of the held-out site: cumulative average over forward passes of
                patches drawn uniformly at random from every volume of that site. No label is
                read and no weight is updated.

Every rule is evaluated on the full case list of the held-out site (Dice only).
Output: results/fl_r1/fedbn_heldout/<tag>.json.
Usage: eval_fedbn_heldout.py <tag> [...] | --tags-file F   [--heldout SITE --max-cases N  (smoke test)]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/holiday/lung_ct")
sys.path.insert(0, str(ROOT / "fl_study"))
import train_fl as T  # noqa: E402

RES = ROOT / "results" / "fl"
CK = ROOT / "data" / "fl_ckpt"
OUT = ROOT / "results" / "fl_r1" / "fedbn_heldout"
PATCHES_PER_CASE = 4
BATCH = 4


def uniform_bn(client_bn, keys):
    out = {}
    for k in next(iter(client_bn.values())):
        ts = [client_bn[s][k] for s in keys]
        out[k] = torch.stack([t.to(torch.float64) for t in ts]).mean(0).to(ts[0].dtype) if ts[0].is_floating_point() else ts[0].clone()
    return out


@torch.no_grad()
def reestimate_stats(model, site, device, seed):
    """Running mean / variance of every BN layer from unlabeled patches of `site` (cumulative average)."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    old = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()
    rng = np.random.RandomState(seed)
    buf, n = [], 0

    def flush():
        nonlocal buf, n
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf))[:, None].to(device).contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            model(x)
        n += len(buf); buf = []

    rows = site.split["full"][:T.MAX_EVAL] if T.MAX_EVAL else site.split["full"]
    for r in rows:
        img, _ = site.arrays(r)                       # the mask is not read
        sh = img.shape
        for _ in range(PATCHES_PER_CASE):
            start = [rng.randint(0, max(1, sh[i] - T.PATCH[i] + 1)) for i in range(3)]
            pi = np.asarray(img[tuple(slice(start[i], start[i] + T.PATCH[i]) for i in range(3))])
            pad = [max(0, T.PATCH[i] - pi.shape[i]) for i in range(3)]
            if any(pad):
                pi = np.pad(pi, [(0, p) for p in pad], constant_values=-1000)
            buf.append(T.norm(pi))
            if len(buf) == BATCH:
                flush()
    if len(buf) > 1:                                   # a batch of one is skipped (variance of a single sample)
        flush()
    for m, mom in zip(bns, old):
        m.momentum = mom; m.eval()
    model.eval()
    return n


def process(tag, device, heldout_override=None):
    op = OUT / f"{tag}.json"
    if op.exists() and not heldout_override:
        print(f"{tag}: exists, skipped", flush=True); return
    r = json.load(open(RES / f"{tag}.json"))
    ck = torch.load(CK / f"{tag}.pt", map_location="cpu")
    assert isinstance(ck, dict) and "client_bn" in ck, f"{tag}: not a FedBN checkpoint"
    clients = list(ck["client_bn"])
    held = heldout_override or [s for s in T.SITE_ORDER if s not in r["train_sites"]][0]
    site = T.Site(held, r["seed"])
    model, _ = T.build_model("scratch", 0, device)
    g = ck["global"]
    bnk = set(next(iter(ck["client_bn"].values())))
    rules = {"avg_size": {k: g[k] for k in bnk}, "avg_uniform": uniform_bn(ck["client_bn"], clients)}
    for c in clients:
        rules[f"client_{c}"] = ck["client_bn"][c]
    out = {"tag": tag, "seed": r["seed"], "train_sites": r["train_sites"], "heldout": held, "best_round": r["best"].get("round"),
           "n_bn_tensors": len(bnk), "rules": {}, "smoke": bool(heldout_override)}
    t0 = time.time()
    for name, bn in rules.items():
        model.load_state_dict(T.merged_state(g, bn))
        res = T.evaluate(model, site, "full", device, full=False)
        out["rules"][name] = res
        print(f"  {tag} {held} {name}: n={len(res['dice'])} dice={np.mean(res['dice']):.4f}", flush=True)
    model.load_state_dict(T.merged_state(g, rules["avg_size"]))
    n = reestimate_stats(model, site, device, r["seed"])
    res = T.evaluate(model, site, "full", device, full=False)
    out["rules"]["target_stats"] = res
    out["target_stats_patches"] = n
    print(f"  {tag} {held} target_stats ({n} patches): n={len(res['dice'])} dice={np.mean(res['dice']):.4f}", flush=True)
    stored = r["results"].get(held, {}).get("full")
    if stored:
        ref = dict(zip(stored["ids"], stored["dice"]))
        d = np.array([abs(v - ref[c]) for c, v in zip(out["rules"]["avg_size"]["ids"], out["rules"]["avg_size"]["dice"]) if c in ref])
        out["verify"] = {"n": int(len(d)), "max_abs_dice_diff": float(d.max()), "mean_abs_dice_diff": float(d.mean())}
        # same tolerance as eval_postproc.py (fp16 sliding-window inference is not bit-reproducible)
        assert d.max() < 5e-3 and d.mean() < 5e-4, f"{tag}: avg_size differs from the stored held-out result ({out['verify']})"
    out["time_s"] = time.time() - t0
    if heldout_override:
        print("smoke test only, nothing written:", {k: float(np.mean(v["dice"])) for k, v in out["rules"].items()}, flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w"), indent=1)
    print(f"== {tag}: {out.get('verify')} {out['time_s']/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--tags-file")
    ap.add_argument("--heldout", help="smoke test: treat this site as unseen for a checkpoint that trained on it")
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args()
    tags = list(a.tags)
    if a.tags_file:
        tags += [t.strip() for t in open(a.tags_file) if t.strip() and not t.startswith("#")]
    T.MAX_EVAL = a.max_cases
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    for tag in tags:
        process(tag, device, a.heldout)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
