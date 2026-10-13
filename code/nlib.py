"""nlib.py - training library for the Inter-IIT CV bootcamp (CIFAR-10N, learning with noisy labels).

Everything is trained FROM SCRATCH with the ResNet-18 in the provided model.py.
No external data, no pretrained weights. Clean labels are used ONLY for
 (a) training the Task-1 clean reference model, and
 (b) *measuring* (Task-1 confusion matrix, Task-2 curves, Task-3 self-check) -
they never influence training, filtering, relabelling or checkpoint selection in Tasks 2-5.

Shared recipe (identical for the baseline and every method):
  SGD(momentum 0.9, weight decay 5e-4), lr 0.1, cosine annealing per epoch to ~0,
  batch 128, 60 epochs/network, augmentation = random crop 32 (zero-pad 4) + horizontal flip,
  mixed precision (fp16) on GPU.
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import ResNet18

NUM_CLASSES = 10


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg, fh=None):
    line = time.strftime("%H:%M:%S ") + str(msg)
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def cosine_lr(cfg, epoch):
    """Learning rate for 0-based `epoch` (identical schedule for every network)."""
    return cfg["lr"] * 0.5 * (1.0 + math.cos(math.pi * epoch / cfg["epochs"]))


def acc(pred, y):
    return float((pred == y).mean()) if len(y) else float("nan")


# ----------------------------------------------------------------------------- data
class Data:
    """Holds all images on the GPU as uint8 NHWC; labels as numpy arrays indexed by CIFAR index."""

    def __init__(self, data_dir, device):
        self.device = device
        df = pd.read_csv(os.path.join(data_dir, "labels.csv")).sort_values("index").reset_index(drop=True)
        assert (df["index"].values == np.arange(len(df))).all(), "labels.csv index column is not 0..N-1"
        self.labels = {
            "noisy": df["noisy_label"].values.astype(np.int64),
            "synthetic": df["synthetic_label"].values.astype(np.int64),
            "clean": df["clean_label"].values.astype(np.int64),  # measurement / clean-ref only
        }
        self.train_idx = np.loadtxt(os.path.join(data_dir, "train_idx.txt"), dtype=np.int64).reshape(-1)
        self.val_idx = np.loadtxt(os.path.join(data_dir, "val_idx.txt"), dtype=np.int64).reshape(-1)
        assert len(set(self.train_idx) & set(self.val_idx)) == 0
        imgs = np.load(os.path.join(data_dir, "train_images.npy"))
        test = np.load(os.path.join(data_dir, "test_images.npy"))
        assert imgs.shape[1:] == (32, 32, 3) and imgs.dtype == np.uint8 and len(imgs) == len(df)
        # normalisation statistics from the provided training split only
        sub = imgs[self.train_idx].astype(np.float32) / 255.0
        self.mean = torch.tensor(sub.mean(axis=(0, 1, 2)), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(sub.std(axis=(0, 1, 2)), device=device).view(1, 3, 1, 1)
        del sub
        self.X = torch.from_numpy(imgs).to(device)        # (50000,32,32,3) uint8
        self.Xtest = torch.from_numpy(test).to(device)    # (10000,32,32,3) uint8
        self.n_test = len(test)

    # --- tensors -> network input
    def to_input(self, xb_uint8):
        x = xb_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)

    def augment(self, xb):
        """Random crop 32x32 from a zero-padded (4px) image + random horizontal flip, on the GPU."""
        n = xb.shape[0]
        dev = xb.device
        p = torch.zeros((n, 40, 40, 3), dtype=torch.uint8, device=dev)
        p[:, 4:36, 4:36] = xb
        ox = torch.randint(0, 9, (n,), device=dev)
        oy = torch.randint(0, 9, (n,), device=dev)
        ar = torch.arange(32, device=dev)
        flip = torch.rand(n, device=dev) < 0.5
        cols = torch.where(flip[:, None], 31 - ar[None, :], ar[None, :]) + ox[:, None]
        rows = ar[None, :] + oy[:, None]
        out = p[torch.arange(n, device=dev)[:, None, None], rows[:, :, None], cols[:, None, :]]
        return self.to_input(out)


def new_net(seed, device):
    set_seed(seed)
    net = ResNet18(num_classes=NUM_CLASSES).to(device)
    return net.to(memory_format=torch.channels_last)


def make_opt(net, cfg):
    return torch.optim.SGD(net.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                           weight_decay=cfg["weight_decay"])


@torch.no_grad()
def predict_logits(nets, data, ids=None, test=False, bs=1024, amp=True, tta=False):
    """Average softmax of one or several nets (eval mode, no augmentation).
    tta=True also averages over the horizontally flipped image (test-time augmentation; used for val/test predictions).
    Returns (probs, logits_of_first_net_on_unflipped_images)."""
    if not isinstance(nets, (list, tuple)):
        nets = [nets]
    for n in nets:
        n.eval()
    src = data.Xtest if test else data.X
    if ids is None:
        ids_t = torch.arange(src.shape[0], device=data.device)
    else:
        ids_t = torch.as_tensor(ids, device=data.device)
    probs, logits0 = [], []
    for i in range(0, len(ids_t), bs):
        x = data.to_input(src[ids_t[i:i + bs]])
        views = [x, x.flip(3).contiguous(memory_format=torch.channels_last)] if tta else [x]
        ps = []
        for vi, v in enumerate(views):
            with autocast(data.device, amp):
                outs = [n(v).float() for n in nets]
            ps += [F.softmax(o, 1) for o in outs]
            if vi == 0:
                logits0.append(outs[0].cpu())
        probs.append(torch.stack(ps).mean(0).cpu())
    return torch.cat(probs).numpy(), torch.cat(logits0).numpy()


# ----------------------------------------------------------------------------- losses
def loss_ce(logits, y, **_):
    return F.cross_entropy(logits, y)


def loss_gce(logits, y, q=0.7, **_):
    """Generalized Cross Entropy (Zhang & Sabuncu, 2018)."""
    p = F.softmax(logits, 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
    return ((1.0 - p ** q) / q).mean()


class ELR:
    """Early-Learning Regularisation (Liu et al., NeurIPS 2020). CIFAR-10 defaults: beta=0.7, lambda=3."""

    def __init__(self, n_total, device, beta=0.7, lam=3.0):
        self.t = torch.zeros((n_total, NUM_CLASSES), device=device)
        self.beta, self.lam = beta, lam

    def __call__(self, logits, y, ids=None, **_):
        p = F.softmax(logits, 1).clamp(1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            pd_ = p / p.sum(1, keepdim=True)
            self.t[ids] = self.beta * self.t[ids] + (1.0 - self.beta) * pd_
        reg = torch.log(1.0 - (self.t[ids] * p).sum(1)).mean()
        return F.cross_entropy(logits, y) + self.lam * reg

    @torch.no_grad()
    def update(self, ids, probs):
        p = probs.clamp(1e-4, 1.0 - 1e-4)
        self.t[ids] = self.beta * self.t[ids] + (1.0 - self.beta) * (p / p.sum(1, keepdim=True))

    def mix_loss(self, logits, y, ids, lam, perm):
        """ELR on a mixup batch (as in ELR+): mixed CE + ELR term against the mixed running targets."""
        p = F.softmax(logits, 1).clamp(1e-4, 1.0 - 1e-4)
        ce = lam * F.cross_entropy(logits, y) + (1 - lam) * F.cross_entropy(logits, y[perm])
        tm = lam * self.t[ids] + (1 - lam) * self.t[ids[perm]]
        reg = torch.log(1.0 - (tm * p).sum(1)).mean()
        return ce + self.lam * reg


# ----------------------------------------------------------------------------- evaluation helpers
def val_metrics(probs_val, data, sel_key):
    pred = probs_val.argmax(1)
    vi = data.val_idx
    m = {"val_acc_noisy": acc(pred, data.labels["noisy"][vi]),
         "val_acc_synthetic": acc(pred, data.labels["synthetic"][vi]),
         "val_acc_clean": acc(pred, data.labels["clean"][vi])}   # clean: logged for analysis only
    m["val_acc_sel"] = m["val_acc_" + sel_key]
    return m


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# ----------------------------------------------------------------------------- single/ensemble training
def run_single(data, cfg, out_dir, method, label_key, train_ids, seed,
               dynamics=False, heldout_ids=None, fh=None, n_nets=1):
    """Train `n_nets` ResNet-18s (seeds seed, seed+1) for cfg['epochs'] epochs; predictions average their softmax.

    method     : 'ce' | 'gce' | 'elr' | 'elr_mix' (ELR + mixup, as in ELR+)
    label_key  : which label column is used for TRAINING ('clean' only for the Task-1 reference)
    Checkpoint selection uses noisy val labels (or synthetic val labels when training on synthetic).
    Val/test predictions use horizontal-flip test-time augmentation.
    dynamics   : log Task-2 curves + per-sample margins/losses on the training set (eval mode, no aug)
    heldout_ids: Task-3 cross-validation - store held-out probabilities at the best (noisy-val) epoch
    """
    dev = data.device
    amp = cfg["amp"] and dev.type == "cuda"
    sel_key = "synthetic" if label_key == "synthetic" else "noisy"
    y_all = torch.from_numpy(data.labels[label_key]).to(dev)
    tr = torch.as_tensor(train_ids, device=dev)
    n, B, E = len(tr), cfg["batch_size"], cfg["epochs"]
    assert not (dynamics and n_nets != 1)
    mix_alpha = cfg.get("mixup_alpha", 1.0)

    nets = [new_net(seed + k, dev) for k in range(n_nets)]
    opts = [make_opt(nt, cfg) for nt in nets]
    scalers = [make_scaler(amp) for _ in nets]
    if method == "ce":
        crits = [loss_ce] * n_nets
    elif method == "gce":
        crits = [loss_gce] * n_nets
    elif method in ("elr", "elr_mix"):
        crits = [ELR(len(data.labels["noisy"]), dev) for _ in nets]
    else:
        raise ValueError(method)

    hist, best = [], {"val_acc_sel": -1.0}
    if dynamics:
        y_noisy_tr = data.labels[label_key][train_ids]
        mislab = data.labels[label_key][train_ids] != data.labels["clean"][train_ids]  # measurement only
        margins = np.zeros((E, n), dtype=np.float16)
        losses = np.zeros((E, n), dtype=np.float16)
    heldout_best = None
    t0 = time.time()
    for ep in range(E):
        lr = cosine_lr(cfg, ep)
        for k in range(n_nets):
            net, opt, scaler, crit = nets[k], opts[k], scalers[k], crits[k]
            for g in opt.param_groups:
                g["lr"] = lr
            net.train()
            perm_all = tr[torch.randperm(n, device=dev)]
            for i in range(0, n - B + 1, B):                       # drop_last
                ids = perm_all[i:i + B]
                x = data.augment(data.X[ids])
                y = y_all[ids]
                if method == "elr_mix":
                    with torch.no_grad():
                        with autocast(dev, amp):
                            p_clean = F.softmax(net(x).float(), 1)
                    crit.update(ids, p_clean)                       # ELR targets from the un-mixed view
                    lam = float(np.random.beta(mix_alpha, mix_alpha))
                    lam = max(lam, 1.0 - lam)
                    pm = torch.randperm(B, device=dev)
                    xm = (lam * x + (1.0 - lam) * x[pm]).contiguous(memory_format=torch.channels_last)
                    with autocast(dev, amp):
                        logits = net(xm)
                    loss = crit.mix_loss(logits.float(), y, ids, lam, pm)
                else:
                    with autocast(dev, amp):
                        logits = net(x)
                    loss = crit(logits.float(), y, ids=ids)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        # ---- end of epoch evaluation (ensemble of all nets, flip TTA)
        pv, _ = predict_logits(nets, data, data.val_idx, amp=amp, tta=True)
        m = val_metrics(pv, data, sel_key)
        m.update(epoch=ep + 1, lr=lr)
        pt, _ = predict_logits(nets, data, test=True, amp=amp, tta=True)
        test_pred = pt.argmax(1)
        if m["val_acc_sel"] > best["val_acc_sel"]:              # earliest best (strict >)
            best = dict(m)
            np.save(os.path.join(out_dir, "test_pred_best.npy"), test_pred)
            if heldout_ids is not None:
                ph, _ = predict_logits(nets, data, heldout_ids, amp=amp, tta=True)
                heldout_best = ph
        if dynamics:
            ptr, ltr = predict_logits(nets[0], data, train_ids, amp=amp)
            pred_tr = ptr.argmax(1)
            m["train_acc_correct"] = acc(pred_tr[~mislab], y_noisy_tr[~mislab])
            m["train_acc_mislabeled"] = acc(pred_tr[mislab], y_noisy_tr[mislab])  # fit to the WRONG label
            m["train_acc_mislabeled_vs_clean"] = acc(pred_tr[mislab], data.labels["clean"][train_ids][mislab])
            ly = ltr[np.arange(n), y_noisy_tr]
            other = ltr.copy()
            other[np.arange(n), y_noisy_tr] = -np.inf
            margins[ep] = (ly - other.max(1)).astype(np.float16)
            losses[ep] = (-np.log(np.clip(ptr[np.arange(n), y_noisy_tr], 1e-12, 1))).astype(np.float16)
        hist.append(m)
        extra = (f" | tr_correct {m['train_acc_correct']:.4f} tr_mislab {m['train_acc_mislabeled']:.4f}"
                 if dynamics else "")
        log(f"[{method}x{n_nets}/{label_key}] ep {ep + 1:2d}/{E} lr {lr:.4f} "
            f"val_noisy {m['val_acc_noisy']:.4f} val_synth {m['val_acc_synthetic']:.4f} "
            f"val_clean {m['val_acc_clean']:.4f}{extra} ({time.time() - t0:.0f}s)", fh)
    np.save(os.path.join(out_dir, "test_pred_final.npy"), test_pred)
    if dynamics:
        np.save(os.path.join(out_dir, "train_margins.npy"), margins)
        np.save(os.path.join(out_dir, "train_losses.npy"), losses)
    if heldout_ids is not None:
        np.save(os.path.join(out_dir, "heldout_ids.npy"), np.asarray(heldout_ids))
        np.save(os.path.join(out_dir, "heldout_probs_best.npy"), heldout_best)
    res = {"method": method, "n_nets": n_nets, "label_key": label_key, "sel_key": sel_key, "epochs": E,
           "history": hist, "best": best, "final": hist[-1], "seconds": time.time() - t0}
    save_json(res, os.path.join(out_dir, "result.json"))
    return res


# ----------------------------------------------------------------------------- DivideMix (2 networks)
@torch.no_grad()
def per_sample_loss(net, data, ids, y_np, amp):
    p, _ = predict_logits(net, data, ids, amp=amp)
    return -np.log(np.clip(p[np.arange(len(ids)), y_np], 1e-12, 1.0))


def gmm_clean_prob(loss):
    from sklearn.mixture import GaussianMixture
    l = (loss - loss.min()) / (loss.max() - loss.min() + 1e-12)
    l = l.reshape(-1, 1)
    g = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=0)
    g.fit(l)
    return g.predict_proba(l)[:, int(np.argmin(g.means_.reshape(-1)))]


def run_divmix(data, cfg, out_dir, label_key, train_ids, seed, fh=None):
    """DivideMix (Li, Socher & Hoi, ICLR 2020) with exactly two networks, each trained for cfg['epochs'] epochs.
    Uses the shared optimiser / schedule / augmentation. Checkpoint selection on noisy (or synthetic) val labels."""
    dev = data.device
    amp = cfg["amp"] and dev.type == "cuda"
    dm = cfg["divmix"]
    warm, T, alpha, lam_u, p_thr = dm["warmup"], dm["T"], dm["alpha"], dm["lambda_u"], dm["p_threshold"]
    sel_key = "synthetic" if label_key == "synthetic" else "noisy"
    y_np_all = data.labels[label_key]
    y_all = torch.from_numpy(y_np_all).to(dev)
    tr_np = np.asarray(train_ids)
    tr = torch.as_tensor(tr_np, device=dev)
    y_tr_np = y_np_all[tr_np]
    n, B, E = len(tr), cfg["batch_size"], cfg["epochs"]
    nets = [new_net(seed, dev), new_net(seed + 1, dev)]
    opts = [make_opt(nn_, cfg) for nn_ in nets]
    scalers = [make_scaler(amp), make_scaler(amp)]
    rng = np.random.default_rng(seed)
    hist, best = [], {"val_acc_sel": -1.0}
    t0 = time.time()

    def warmup_epoch(k):
        net, opt, scaler = nets[k], opts[k], scalers[k]
        net.train()
        perm = tr[torch.randperm(n, device=dev)]
        for i in range(0, n - B + 1, B):
            ids = perm[i:i + B]
            with autocast(dev, amp):
                logits = net(data.augment(data.X[ids]))
            logits = logits.float()
            p = F.softmax(logits, 1)
            # CE + confidence penalty (negative entropy), as DivideMix does for asymmetric/structured noise
            loss = F.cross_entropy(logits, y_all[ids]) + (p * torch.log(p.clamp_min(1e-12))).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    def train_epoch(k, prob, ep):
        """Train nets[k] on the division `prob` produced by the OTHER network."""
        net, net2, opt, scaler = nets[k], nets[1 - k], opts[k], scalers[k]
        net.train()
        net2.eval()
        lab_mask = prob > p_thr
        lab = tr_np[lab_mask]
        unl = tr_np[~lab_mask]
        w_all = torch.zeros(len(y_np_all), device=dev)
        w_all[tr] = torch.as_tensor(prob, dtype=torch.float32, device=dev)
        if len(lab) < B:   # degenerate split (should not happen) -> sample with replacement
            lab = rng.choice(tr_np, size=B, replace=True)
        if len(unl) < 2:
            unl = rng.choice(tr_np, size=B, replace=True)
        lab = rng.permutation(lab)
        num_iter = len(lab) // B
        reps = int(math.ceil(num_iter * B / len(unl)))
        unl_seq = np.concatenate([rng.permutation(unl) for _ in range(reps)])[:num_iter * B]
        lab_t = torch.as_tensor(lab, device=dev)
        unl_t = torch.as_tensor(unl_seq, device=dev)
        for it in range(num_iter):
            bx = lab_t[it * B:(it + 1) * B]
            bu = unl_t[it * B:(it + 1) * B]
            xr, ur = data.X[bx], data.X[bu]
            x1, x2, u1, u2 = data.augment(xr), data.augment(xr), data.augment(ur), data.augment(ur)
            yb = F.one_hot(y_all[bx], NUM_CLASSES).float()
            wb = w_all[bx].view(-1, 1)
            with torch.no_grad():
                with autocast(dev, amp):
                    ou11, ou12 = net(u1).float(), net(u2).float()
                    ou21, ou22 = net2(u1).float(), net2(u2).float()
                    ox1, ox2 = net(x1).float(), net(x2).float()
                pu = (F.softmax(ou11, 1) + F.softmax(ou12, 1) + F.softmax(ou21, 1) + F.softmax(ou22, 1)) / 4
                ptu = pu ** (1.0 / T)
                tu = ptu / ptu.sum(1, keepdim=True)                     # co-guessing
                px = (F.softmax(ox1, 1) + F.softmax(ox2, 1)) / 2
                px = wb * yb + (1 - wb) * px                             # label refinement
                ptx = px ** (1.0 / T)
                tx = ptx / ptx.sum(1, keepdim=True)
            l = float(np.random.beta(alpha, alpha))
            l = max(l, 1 - l)
            all_in = torch.cat([x1, x2, u1, u2], 0)
            all_t = torch.cat([tx, tx, tu, tu], 0)
            idx = torch.randperm(all_in.size(0), device=dev)
            mixed_in = (l * all_in + (1 - l) * all_in[idx]).contiguous(memory_format=torch.channels_last)
            mixed_t = l * all_t + (1 - l) * all_t[idx]
            with autocast(dev, amp):
                logits = net(mixed_in)
            logits = logits.float()
            lx, lu = logits[:2 * B], logits[2 * B:]
            Lx = -torch.mean(torch.sum(F.log_softmax(lx, 1) * mixed_t[:2 * B], 1))
            Lu = torch.mean((F.softmax(lu, 1) - mixed_t[2 * B:]) ** 2)
            ramp = float(np.clip((ep + it / num_iter - warm) / 16.0, 0.0, 1.0))
            prior = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES, device=dev)
            pred_mean = F.softmax(logits, 1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean.clamp_min(1e-12)))
            loss = Lx + lam_u * ramp * Lu + penalty
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        return int(lab_mask.sum())

    for ep in range(E):
        lr = cosine_lr(cfg, ep)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr
        n_lab = [None, None]
        if ep < warm:
            warmup_epoch(0)
            warmup_epoch(1)
        else:
            probs = [gmm_clean_prob(per_sample_loss(nets[k], data, tr_np, y_tr_np, amp)) for k in (0, 1)]
            n_lab[0] = train_epoch(0, probs[1], ep)   # net1 uses net2's division
            n_lab[1] = train_epoch(1, probs[0], ep)   # net2 uses net1's division
        pv, _ = predict_logits(nets, data, data.val_idx, amp=amp, tta=True)
        m = val_metrics(pv, data, sel_key)
        m.update(epoch=ep + 1, lr=lr, n_labeled=n_lab)
        pt, _ = predict_logits(nets, data, test=True, amp=amp, tta=True)
        test_pred = pt.argmax(1)
        if m["val_acc_sel"] > best["val_acc_sel"]:
            best = dict(m)
            np.save(os.path.join(out_dir, "test_pred_best.npy"), test_pred)
        hist.append(m)
        log(f"[divmix/{label_key}] ep {ep + 1:2d}/{E} lr {lr:.4f} labeled {n_lab} "
            f"val_noisy {m['val_acc_noisy']:.4f} val_synth {m['val_acc_synthetic']:.4f} "
            f"val_clean {m['val_acc_clean']:.4f} ({time.time() - t0:.0f}s)", fh)
    np.save(os.path.join(out_dir, "test_pred_final.npy"), test_pred)
    # Task-3 signal: per-sample loss of the final networks w.r.t. the (noisy) training label
    ls = []
    for k in (0, 1):
        l_ = per_sample_loss(nets[k], data, tr_np, y_tr_np, amp)
        ls.append((l_ - l_.min()) / (l_.max() - l_.min() + 1e-12))
    np.save(os.path.join(out_dir, "train_loss_score.npy"), ((ls[0] + ls[1]) / 2).astype(np.float32))
    np.save(os.path.join(out_dir, "train_ids.npy"), tr_np)
    res = {"method": "divmix", "label_key": label_key, "sel_key": sel_key, "epochs": E,
           "history": hist, "best": best, "final": hist[-1], "seconds": time.time() - t0}
    save_json(res, os.path.join(out_dir, "result.json"))
    return res
