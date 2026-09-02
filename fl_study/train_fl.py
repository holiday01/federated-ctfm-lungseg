#!/usr/bin/env python3
"""Study 3 trainer: local, centralised and federated fine-tuning of the CT-FM SegResNet for
3D lung-tumour segmentation across three sites (TW = in-house, NL = NSCLC-Radiomics,
US = NSCLC Radiogenomics). One process, one GPU; federated learning is simulated cross-silo
(every client trains every round, sequentially, on the same device).

Modes
  local    --sites S            train on S.train, select on S.val
  central  --sites A,B[,C]      pool the listed sites' train splits, select on the mean of their val Dice
  fedavg   --sites A,B[,C]      FedAvg over the listed sites as clients (--fedprox-mu > 0 -> FedProx,
                                --client-weighting uniform -> unweighted average,
                                --fedbn -> FedBN: BatchNorm affine parameters and running statistics
                                stay local to each client; only the remaining tensors are averaged)

Init: --init ctfm (encoder weights from project-lighter/ct_fm_segresnet, decoder random) or scratch.
Every run evaluates the selected checkpoint on: val + test split of every training site, and the
FULL case list of every site that did not take part in training (held-out-site transfer).
Writes results/fl/<tag>.json (per-case metrics, history, config) and data/fl_ckpt/<tag>.pt.
"""
from __future__ import annotations
import argparse, copy, json, math, os, queue, sys, threading, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path("/home/holiday/lung_ct")
CACHE = ROOT / "data" / "fl_cache"
RESULTS = ROOT / "results" / "fl"
CKPT = ROOT / "data" / "fl_ckpt"
PATCH = (96, 96, 96)
SITE_ORDER = ["TW", "NL", "US"]
# CT-FM intensity convention (model card): HU -1024..2048 -> 0..1. Stored volumes are int16 HU clipped to [-1000, 400].
CTFM_A, CTFM_B = -1024.0, 2048.0


def norm(img):
    return (np.clip((img.astype(np.float32) - CTFM_A) / (CTFM_B - CTFM_A), 0.0, 1.0)).astype(np.float32)


def seed_everything(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
class Site:
    """One institution: memory-mapped volumes + a seeded patient-level train/val/test split."""

    def __init__(self, key, seed, max_fg=4000):
        self.key = key
        self.rows = json.load(open(CACHE / key / "manifest.json"))
        pats = sorted({r["patient"] for r in self.rows})
        rng = np.random.RandomState(seed)
        rng.shuffle(pats)
        n = len(pats); n_test = max(1, n // 5); n_val = max(1, n // 5)
        test_p, val_p = set(pats[:n_test]), set(pats[n_test:n_test + n_val])
        self.split = {"train": [r for r in self.rows if r["patient"] not in test_p | val_p],
                      "val": [r for r in self.rows if r["patient"] in val_p],
                      "test": [r for r in self.rows if r["patient"] in test_p],
                      "full": list(self.rows)}
        self.fg = {}
        self.max_fg = max_fg
        self.rng = np.random.RandomState(seed + 17)

    def arrays(self, r):
        return (np.load(CACHE / self.key / f"{r['id']}_img.npy", mmap_mode="r"),
                np.load(CACHE / self.key / f"{r['id']}_msk.npy", mmap_mode="r"))

    def fg_coords(self, r):
        if r["id"] not in self.fg:
            msk = np.load(CACHE / self.key / f"{r['id']}_msk.npy")
            fg = np.argwhere(msk > 0)
            if len(fg) > self.max_fg:
                fg = fg[self.rng.choice(len(fg), self.max_fg, replace=False)]
            self.fg[r["id"]] = fg.astype(np.int32)
        return self.fg[r["id"]]

    def describe(self):
        return {k: [r["id"] for r in v] for k, v in self.split.items() if k != "full"}


def rand_patch(img, msk, fg, rng, fg_prob=0.8):
    sh = img.shape
    if len(fg) and rng.random() < fg_prob:
        c = fg[rng.randint(len(fg))]
        start = [int(np.clip(c[i] - PATCH[i] // 2, 0, max(0, sh[i] - PATCH[i]))) for i in range(3)]
    else:
        start = [rng.randint(0, max(1, sh[i] - PATCH[i] + 1)) for i in range(3)]
    sl = tuple(slice(start[i], start[i] + PATCH[i]) for i in range(3))
    pi, pm = np.asarray(img[sl]), np.asarray(msk[sl])
    pad = [max(0, PATCH[i] - pi.shape[i]) for i in range(3)]
    if any(pad):
        pi = np.pad(pi, [(0, p) for p in pad], constant_values=-1000)
        pm = np.pad(pm, [(0, p) for p in pad], constant_values=0)
    return norm(pi), pm.astype(np.uint8)


def np_augment(img, msk, rng):
    """Study 2's on-patch augmentation (scripts/train_seg.py np_augment), unchanged: flips, rot90,
    gamma, intensity scale/shift, Gaussian noise. img is the normalised patch in [0, 1]."""
    for ax in range(3):
        if rng.random() < 0.5:
            img = np.flip(img, ax); msk = np.flip(msk, ax)
    if rng.random() < 0.5:
        ax = tuple(rng.choice(3, 2, replace=False)); k = rng.randint(1, 4)
        img = np.rot90(img, k, ax); msk = np.rot90(msk, k, ax)
    img = np.ascontiguousarray(img); msk = np.ascontiguousarray(msk)
    if rng.random() < 0.3:
        g = rng.uniform(0.7, 1.5); img = np.clip(img, 0, None) ** g
    if rng.random() < 0.3:
        img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.2:
        img = img + rng.normal(0, 0.03, img.shape).astype(img.dtype)
    return img.astype(np.float32), msk.astype(np.uint8)


class Sampler:
    """Patch sampler for a list of (site, row) training cases, with a prefetch thread.
    Default: uniform over the pooled volumes. With `groups` (one list per site): a site is
    sampled uniformly first, then a volume within it (site-balanced sampling)."""

    def __init__(self, items, batch, seed, augment=True, prefetch=6, groups=None):
        self.items, self.batch, self.augment = items, batch, augment
        self.groups = groups
        self.rng = np.random.RandomState(seed)
        self.q = queue.Queue(maxsize=prefetch)
        self.stop = False
        self.t = threading.Thread(target=self._worker, daemon=True); self.t.start()

    def _one(self):
        xs, ys = [], []
        for _ in range(self.batch):
            if self.groups is not None:
                g = self.groups[self.rng.randint(len(self.groups))]
                site, r = g[self.rng.randint(len(g))]
            else:
                site, r = self.items[self.rng.randint(len(self.items))]
            img, msk = site.arrays(r)
            pi, pm = rand_patch(img, msk, site.fg_coords(r), self.rng)
            if self.augment:
                pi, pm = np_augment(pi, pm, self.rng)
            xs.append(pi); ys.append(pm)
        return np.stack(xs)[:, None], np.stack(ys).astype(np.int64)[:, None]

    def _worker(self):
        while not self.stop:
            self.q.put(self._one())

    def next(self, device):
        x, y = self.q.get()
        return (torch.from_numpy(x).to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d),
                torch.from_numpy(y).to(device, non_blocking=True))

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


# ----------------------------------------------------------------------------- model
def build_model(init, seed, device):
    from monai.networks.nets import SegResNetDS
    torch.manual_seed(seed)
    net = SegResNetDS(spatial_dims=3, init_filters=32, in_channels=1, out_channels=2,
                      blocks_down=(1, 2, 2, 4, 4), norm="batch", act="relu", dsdepth=1)
    info = {"n_params": int(sum(p.numel() for p in net.parameters()))}
    if init == "ctfm":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download("project-lighter/ct_fm_segresnet", "model.safetensors"))
        enc = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        own = {k for k in net.state_dict() if k.startswith("encoder.")}
        assert set(enc) == own, f"encoder key mismatch: {len(set(enc) ^ own)} keys differ"
        missing, unexpected = net.load_state_dict(enc, strict=False)
        assert not unexpected and all(not k.startswith("encoder.") for k in missing)
        info["ctfm_encoder_keys"] = len(enc)
        info["n_params_encoder"] = int(sum(v.numel() for v in enc.values()))
    # channels_last_3d: on this Blackwell GPU cuDNN's NCDHW fp16 3D kernels are ~20x slower (1.4 vs 0.06 s/iter)
    return net.to(device).to(memory_format=torch.channels_last_3d), info


def sd_to_cpu(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def average_states(states, weights):
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    out = {}
    for k in states[0]:
        t0 = states[0][k]
        if t0.is_floating_point():
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for s, wi in zip(states, w):
                acc += s[k].to(torch.float64) * float(wi)
            out[k] = acc.to(t0.dtype)
        else:
            out[k] = t0.clone()   # num_batches_tracked: unused with fixed BN momentum
    return out


def bn_state_keys(model):
    """state_dict keys that belong to BatchNorm modules (affine weight/bias, running stats, counters)."""
    keys = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            keys.update(f"{name}.{k}" for k in mod.state_dict())
    return keys


def merged_state(global_sd, client_bn=None):
    """Global tensors overlaid with one client's BatchNorm state (FedBN); plain global otherwise."""
    return {**global_sd, **client_bn} if client_bn else global_sd


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict_prob(model, img, device):
    from monai.inferers import sliding_window_inference
    model.eval()
    x = torch.from_numpy(norm(np.asarray(img)))[None, None].to(device).contiguous(memory_format=torch.channels_last_3d)
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        logits = sliding_window_inference(x, PATCH, 4, model, overlap=0.5, mode="gaussian")
    return torch.softmax(logits.float(), 1)[0, 1].cpu().numpy()


METRICS_VERSION = 2   # 2: HD95 = max of the two directed 95th percentiles (MONAI / Metrics Reloaded convention)


def hd95_mm(pred, gt, spacing=1.0):
    """HD95 in mm: maximum over the two directions of the 95th percentile of boundary distances
    (MONAI compute_hausdorff_distance, percentile=95). NaN when either mask is empty. Computed on
    the union bounding box (+5 voxels) for speed; the result is identical because MONAI crops too."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    from monai.metrics import compute_hausdorff_distance
    nz = np.argwhere(pred | gt); lo = np.maximum(nz.min(0) - 5, 0); hi = np.minimum(nz.max(0) + 6, pred.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    p = torch.from_numpy(np.ascontiguousarray(pred[sl]))[None, None].float()
    g = torch.from_numpy(np.ascontiguousarray(gt[sl]))[None, None].float()
    v = float(compute_hausdorff_distance(p, g, include_background=True, percentile=95)[0, 0])
    return v * spacing if np.isfinite(v) else float("nan")


def ece_band(prob, gt, n_bins=10, band=10):
    """Foreground-probability expected calibration error inside (gt dilated by `band` voxels) | prediction."""
    region = ndimage.binary_dilation(gt, iterations=band) | (prob >= 0.5)
    p = prob[region]; y = gt[region].astype(np.float32)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def drop_specks(m, min_vox=10):
    """Remove connected components smaller than min_vox voxels (annotation speckle; TW masks carry
    thousands of single-voxel specks). Used only for the surface metric; Dice uses the raw mask."""
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (sizes >= min_vox)[lab]


def case_metrics(prob, msk, full=True):
    gt = np.asarray(msk) > 0; pred = prob >= 0.5
    inter = float((pred & gt).sum()); denom = float(pred.sum() + gt.sum())
    out = {"dice": 2 * inter / denom if denom > 0 else 0.0, "gt_cm3": float(gt.sum()) / 1000.0, "pred_cm3": float(pred.sum()) / 1000.0}
    if full:
        gt_c = drop_specks(gt)
        out["hd95"] = hd95_mm(pred, gt_c)
        out["ece"] = ece_band(prob, gt)
        out["detected"] = bool(out["dice"] > 0.1)
        lab, n = ndimage.label(pred); out["n_pred_components"] = int(n)
        # largest-connected-component post-processing of the prediction (secondary analysis)
        if n > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            pred_l = lab == int(np.argmax(sizes))
        else:
            pred_l = pred
        inter_l = float((pred_l & gt).sum()); denom_l = float(pred_l.sum() + gt.sum())
        out["dice_lcc"] = 2 * inter_l / denom_l if denom_l > 0 else 0.0
        out["hd95_lcc"] = hd95_mm(pred_l, gt_c)
    return out


MAX_EVAL = 0


def evaluate(model, site, split, device, full=False):
    rows = site.split[split][:MAX_EVAL] if MAX_EVAL else site.split[split]
    res = {"ids": [], "dice": []}
    for r in rows:
        img, msk = site.arrays(r)
        if np.asarray(msk).sum() == 0:
            continue
        m = case_metrics(predict_prob(model, img, device), msk, full=full)
        res["ids"].append(r["id"])
        for k, v in m.items():
            res.setdefault(k, []).append(v)
    return res


def val_score(model, sites, device):
    """Unweighted mean over training sites of the mean per-case val Dice."""
    per = {s.key: float(np.mean(evaluate(model, s, "val", device)["dice"])) for s in sites}
    return float(np.mean(list(per.values()))), per


# ----------------------------------------------------------------------------- training
def make_opt(model, lr, wd, total_iters, warmup, start=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    def f(it):
        it = it + start
        if it < warmup:
            return (it + 1) / warmup
        t = (it - warmup) / max(1, total_iters - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    return opt, sched


def train_iters(model, opt, sched, scaler, sampler, n_iters, loss_fn, device, log_every=100, prox=None, mu=0.0, tag=""):
    model.train()
    t0, run = time.time(), []
    for it in range(n_iters):
        xb, yb = sampler.next(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            loss = loss_fn(model(xb), yb)
        if prox is not None and mu > 0:
            pl = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), prox))
            loss = loss + 0.5 * mu * pl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        run.append(loss.item())
        if (it + 1) % log_every == 0:
            print(f"  {tag} it {it+1}/{n_iters} loss={np.mean(run[-log_every:]):.4f} lr={sched.get_last_lr()[0]:.2e} ({(time.time()-t0)/(it+1):.3f}s/it)", flush=True)
    return float(np.mean(run)) if run else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "central", "fedavg"], required=True)
    ap.add_argument("--sites", required=True, help="comma list from TW,NL,US")
    ap.add_argument("--init", choices=["ctfm", "scratch"], default="ctfm")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters-per-site", type=int, default=6000, help="gradient steps per participating site")
    ap.add_argument("--eval-every", type=int, default=1000, help="local/central: validation interval (iterations)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-iters", type=int, default=200)
    ap.add_argument("--eval-every-rounds", type=int, default=2)
    ap.add_argument("--fedprox-mu", type=float, default=0.0)
    ap.add_argument("--client-weighting", choices=["size", "uniform"], default="size")
    ap.add_argument("--fedbn", action="store_true", help="fedavg: keep BatchNorm parameters/statistics local to each client (FedBN)")
    ap.add_argument("--reset-client-opt", action="store_true", help="fedavg: rebuild each client's AdamW state at every round (LR schedule position kept)")
    ap.add_argument("--site-balanced", action="store_true", help="central: sample a site uniformly, then a volume within it (equal expected per-site steps)")
    ap.add_argument("--personalize-iters", type=int, default=0, help="fedavg: per-client fine-tuning of the selected global model")
    ap.add_argument("--personalize-lr", type=float, default=1e-4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-heldout-eval", action="store_true", help="skip full-site evaluation of non-training sites")
    ap.add_argument("--max-eval-cases", type=int, default=0, help="debug: cap cases per evaluated split (0 = all)")
    cfg = ap.parse_args()

    seed_everything(cfg.seed)
    global MAX_EVAL; MAX_EVAL = cfg.max_eval_cases
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_keys = [k for k in SITE_ORDER if k in cfg.sites.split(",")]
    assert train_keys, "no valid sites"
    all_sites = {k: Site(k, cfg.seed) for k in SITE_ORDER if (CACHE / k / "manifest.json").exists()}
    sites = [all_sites[k] for k in train_keys]
    variant = ""
    if cfg.mode == "central" and cfg.site_balanced:
        variant = "central-balanced"
    if cfg.mode == "fedavg":
        variant = ("fedprox%g" % cfg.fedprox_mu if cfg.fedprox_mu > 0 else ("fedbn" if cfg.fedbn else "fedavg")) + ("-uniform" if cfg.client_weighting == "uniform" else "") + ("-optreset" if cfg.reset_client_opt else "")
        assert not (cfg.fedbn and cfg.fedprox_mu > 0), "fedbn and fedprox are separate ablations"
    tag = cfg.tag or f"{variant or cfg.mode}_{'-'.join(train_keys)}_{cfg.init}_s{cfg.seed}"
    RESULTS.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    print(f"=== {tag} ({ts}) mode={cfg.mode} sites={train_keys} init={cfg.init} seed={cfg.seed}", flush=True)
    for s in sites:
        print(f"  {s.key}: train={len(s.split['train'])} val={len(s.split['val'])} test={len(s.split['test'])} volumes "
              f"({len({r['patient'] for r in s.split['train']})} train patients)", flush=True)

    from monai.losses import DiceCELoss
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    model, minfo = build_model(cfg.init, cfg.seed, device)
    print(f"  model: {minfo}", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best = [], {"score": -1.0}
    best_path = CKPT / f"{tag}.pt"
    t_train, t_eval, t_start = 0.0, 0.0, time.time()
    comm_bytes = 0

    if cfg.mode in ("local", "central"):
        total = cfg.iters_per_site * len(sites)
        items = [(s, r) for s in sites for r in s.split["train"]]
        groups = [[(s, r) for r in s.split["train"]] for s in sites] if cfg.site_balanced else None
        sampler = Sampler(items, cfg.batch, cfg.seed, augment=not cfg.no_augment, groups=groups)
        opt, sched = make_opt(model, cfg.lr, cfg.wd, total, cfg.warmup)
        done = 0
        while done < total:
            n = min(cfg.eval_every, total - done)
            t0 = time.time()
            loss = train_iters(model, opt, sched, scaler, sampler, n, loss_fn, device, tag=tag)
            t_train += time.time() - t0; done += n
            t0 = time.time(); score, per = val_score(model, sites, device); t_eval += time.time() - t0
            history.append({"iter": done, "loss": loss, "val_score": score, "val": per})
            print(f"[eval] it {done}/{total} val_score={score:.4f} {per}", flush=True)
            if score > best["score"]:
                best = {"score": score, "iter": done, "val": per}; torch.save(model.state_dict(), best_path)
        sampler.close()
    else:  # federated
        K = len(sites)
        n_train = [len(s.split["train"]) for s in sites]
        weights = n_train if cfg.client_weighting == "size" else [1] * K
        total_local = cfg.rounds * cfg.local_iters
        samplers = [Sampler([(s, r) for r in s.split["train"]], cfg.batch, cfg.seed * 100 + i, augment=not cfg.no_augment) for i, s in enumerate(sites)]
        opts = [make_opt(model, cfg.lr, cfg.wd, total_local, cfg.warmup) for _ in sites]
        scalers = [torch.amp.GradScaler("cuda", enabled=device.type == "cuda") for _ in sites]
        global_sd = sd_to_cpu(model)
        bnk = bn_state_keys(model) if cfg.fedbn else set()
        # FedBN: each client keeps its own BatchNorm state; the averaged BN state in global_sd is only
        # used for sites that never trained (held-out evaluation), where no client state exists.
        client_bn = [{k: global_sd[k].clone() for k in bnk} for _ in sites] if cfg.fedbn else [None] * K
        param_bytes = sum(v.numel() * v.element_size() for k, v in global_sd.items() if k not in bnk)
        print(f"  federated tensors: {param_bytes/1e6:.1f} MB/model" + (f" (FedBN keeps {len(bnk)} BN tensors local)" if cfg.fedbn else ""), flush=True)

        def fl_eval(sd):
            per = {}
            for i, s in enumerate(sites):
                model.load_state_dict(merged_state(sd, client_bn[i]))
                per[s.key] = float(np.mean(evaluate(model, s, "val", device)["dice"]))
            return float(np.mean(list(per.values()))), per

        for rd in range(1, cfg.rounds + 1):
            states, losses = [], []
            t0 = time.time()
            for i, s in enumerate(sites):
                model.load_state_dict(merged_state(global_sd, client_bn[i]))
                if cfg.reset_client_opt:
                    opts[i] = make_opt(model, cfg.lr, cfg.wd, total_local, cfg.warmup, start=(rd - 1) * cfg.local_iters)
                prox = [p.detach().clone() for p in model.parameters()] if cfg.fedprox_mu > 0 else None
                loss = train_iters(model, opts[i][0], opts[i][1], scalers[i], samplers[i], cfg.local_iters, loss_fn, device,
                                   log_every=cfg.local_iters, prox=prox, mu=cfg.fedprox_mu, tag=f"r{rd} {s.key}")
                sd = sd_to_cpu(model)
                if cfg.fedbn:
                    client_bn[i] = {k: sd[k] for k in bnk}
                states.append(sd); losses.append(loss)
                del prox
            global_sd = average_states(states, weights)
            comm_bytes += 2 * K * param_bytes
            t_train += time.time() - t0
            rec = {"round": rd, "loss": dict(zip(train_keys, losses))}
            if rd % cfg.eval_every_rounds == 0 or rd == cfg.rounds:
                t0 = time.time(); score, per = fl_eval(global_sd); t_eval += time.time() - t0
                rec.update({"val_score": score, "val": per})
                print(f"[eval] round {rd}/{cfg.rounds} val_score={score:.4f} {per}", flush=True)
                if score > best["score"]:
                    best = {"score": score, "round": rd, "val": per}
                    torch.save({"global": global_sd, "client_bn": {s.key: client_bn[i] for i, s in enumerate(sites)}} if cfg.fedbn else global_sd, best_path)
            history.append(rec)
        for sm in samplers:
            sm.close()

    # ---------------------------------------------------------------- final evaluation of the selected model
    ck = torch.load(best_path, map_location="cpu")
    ck_global, ck_bn = (ck["global"], ck["client_bn"]) if isinstance(ck, dict) and "client_bn" in ck else (ck, {})
    t0 = time.time()
    results = {}
    for k, s in all_sites.items():
        results[k] = {}
        model.load_state_dict(merged_state(ck_global, ck_bn.get(k)))   # FedBN: own BN at training sites, averaged BN elsewhere
        if k in train_keys:
            for split in ("val", "test"):
                results[k][split] = evaluate(model, s, split, device, full=True)
        elif not cfg.no_heldout_eval:
            results[k]["full"] = evaluate(model, s, "full", device, full=True)
        for split, res in results[k].items():
            print(f"  [{tag}] {k}/{split}: n={len(res['dice'])} dice={np.mean(res['dice']):.4f} "
                  f"hd95={np.nanmedian(res['hd95']):.1f}mm ece={np.nanmean(res['ece']):.4f} det={np.mean(res['detected']):.3f}", flush=True)
    personalized = {}
    if cfg.mode == "fedavg" and cfg.personalize_iters > 0:
        for i, s in enumerate(sites):
            model.load_state_dict(merged_state(ck_global, ck_bn.get(s.key)))
            sm = Sampler([(s, r) for r in s.split["train"]], cfg.batch, cfg.seed * 1000 + i, augment=not cfg.no_augment)
            opt, sched = make_opt(model, cfg.personalize_lr, cfg.wd, cfg.personalize_iters, 20)
            sc = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            pbest, pstate, phist, done = -1.0, None, [], 0
            step = max(50, cfg.personalize_iters // 5)
            while done < cfg.personalize_iters:
                n = min(step, cfg.personalize_iters - done)
                train_iters(model, opt, sched, sc, sm, n, loss_fn, device, log_every=n, tag=f"ft {s.key}")
                done += n
                v = float(np.mean(evaluate(model, s, "val", device)["dice"])); phist.append({"iter": done, "val": v})
                if v > pbest:
                    pbest, pstate = v, sd_to_cpu(model)
            sm.close()
            model.load_state_dict(pstate)
            personalized[s.key] = {"val_best": pbest, "history": phist, "test": evaluate(model, s, "test", device, full=True)}
            print(f"  [{tag}] personalised {s.key}: val={pbest:.4f} test dice={np.mean(personalized[s.key]['test']['dice']):.4f}", flush=True)
    t_eval += time.time() - t0

    def summ(res):
        return {"n": len(res["dice"]), "dice_mean": float(np.mean(res["dice"])), "dice_median": float(np.median(res["dice"])),
                "hd95_median": float(np.nanmedian(res["hd95"])), "empty_pred_frac": float(np.mean(np.isnan(res["hd95"]))),
                "ece_mean": float(np.nanmean(res["ece"])), "detection": float(np.mean(res["detected"])),
                "dice_lcc_mean": float(np.mean(res["dice_lcc"])) if "dice_lcc" in res else None,
                "hd95_lcc_median": float(np.nanmedian(res["hd95_lcc"])) if "hd95_lcc" in res else None}
    summary = {k: {sp: summ(r) for sp, r in v.items()} for k, v in results.items()}
    out = {"tag": tag, "timestamp": ts, "mode": cfg.mode, "variant": variant or cfg.mode, "train_sites": train_keys, "init": cfg.init,
           "seed": cfg.seed, "config": vars(cfg), "model": minfo, "splits": {k: s.describe() for k, s in all_sites.items()},
           "history": history, "best": best, "results": results, "summary": summary, "personalized": personalized,
           "comm_bytes": int(comm_bytes), "time_train_s": t_train, "time_eval_s": t_eval, "time_total_s": time.time() - t_start,
           "checkpoint": str(best_path), "checkpoint_format": "fedbn" if cfg.fedbn else "state_dict", "metrics_version": METRICS_VERSION}
    jp = RESULTS / f"{tag}.json"
    json.dump(out, open(jp, "w"), indent=1)
    print(f"=== done {tag}: best={best} summary={json.dumps(summary)}\nwrote {jp}", flush=True)


if __name__ == "__main__":
    main()
