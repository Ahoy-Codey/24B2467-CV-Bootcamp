"""Run one training job.   python run_job.py --job <name> --cfg cfg.json

Jobs (each trains ResNet-18 from scratch, <= 60 epochs per network):
  t1_clean        Task 1  CE on CLEAN labels (reference model)
  t1_noisy        Task 1+2 CE on noisy labels, logs dynamics, early-stop predictions, Task-3 margins
  t3_fold{0,1,2}  Task 3  3-fold cross-validation (CE, noisy labels) -> out-of-fold probabilities
  t4_divmix       Task 4  DivideMix (2 networks)
  t4_elr          Task 4  Early-Learning Regularisation (2-network ensemble)
  t4_elrmix       Task 4  ELR + mixup (as in ELR+), 2-network ensemble
  t5_base_synth   Task 5  CE baseline on synthetic labels
  t5_best_synth   Task 5  best Task-4 method (by noisy val acc) on synthetic labels
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlib  # noqa: E402

T4_METHODS = ["divmix", "elr", "elrmix"]
T4_SPEC = {"elr": ("elr", 2), "elrmix": ("elr_mix", 2)}   # job suffix -> (nlib method, n networks)


def best_t4_method(out_root):
    """Best Task-4 method = highest NOISY validation accuracy (at its selected checkpoint)."""
    scores = {}
    for m in T4_METHODS:
        p = os.path.join(out_root, f"t4_{m}", "result.json")
        if os.path.exists(p):
            with open(p) as f:
                scores[m] = json.load(f)["best"]["val_acc_noisy"]
    if not scores:
        raise RuntimeError("no Task-4 results found")
    return max(scores, key=scores.get), scores


def fold_split(train_idx, k, seed):
    rng = np.random.default_rng(seed + 12345)
    perm = rng.permutation(train_idx)
    return np.array_split(perm, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--cfg", required=True)
    a = ap.parse_args()
    with open(a.cfg) as f:
        cfg = json.load(f)
    out_dir = os.path.join(cfg["out_root"], a.job)
    os.makedirs(out_dir, exist_ok=True)
    done_flag = os.path.join(out_dir, "DONE")
    if os.path.exists(done_flag):
        print("already done:", a.job)
        return
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fh = open(os.path.join(out_dir, "train.log"), "a")
    nlib.log(f"job {a.job} on {dev} "
             f"({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})", fh)
    data = nlib.Data(cfg["data_dir"], dev)
    tr = data.train_idx
    if cfg.get("debug_n_train"):                       # local smoke test only
        tr = tr[:cfg["debug_n_train"]]
    seed = cfg["seed"]
    j = a.job
    if j == "t1_clean":
        nlib.run_single(data, cfg, out_dir, "ce", "clean", tr, seed, fh=fh)
    elif j == "t1_noisy":
        nlib.run_single(data, cfg, out_dir, "ce", "noisy", tr, seed, dynamics=True, fh=fh)
    elif j.startswith("t3_fold"):
        f = int(j[len("t3_fold"):])
        folds = fold_split(tr, cfg["n_folds"], seed)
        held = np.sort(folds[f])
        train_part = np.sort(np.concatenate([folds[i] for i in range(cfg["n_folds"]) if i != f]))
        nlib.run_single(data, cfg, out_dir, "ce", "noisy", train_part, seed, heldout_ids=held, fh=fh)
    elif j == "t4_divmix":
        nlib.run_divmix(data, cfg, out_dir, "noisy", tr, seed, fh=fh)
    elif j[3:] in T4_SPEC:
        meth, k = T4_SPEC[j[3:]]
        nlib.run_single(data, cfg, out_dir, meth, "noisy", tr, seed, fh=fh, n_nets=k)
    elif j == "t5_base_synth":
        nlib.run_single(data, cfg, out_dir, "ce", "synthetic", tr, seed, fh=fh)
    elif j == "t5_best_synth":
        m, scores = best_t4_method(cfg["out_root"])
        nlib.log(f"Task-4 noisy-val accuracies {scores} -> best method '{m}'", fh)
        with open(os.path.join(out_dir, "method.txt"), "w") as fm:
            fm.write(m)
        if m == "divmix":
            nlib.run_divmix(data, cfg, out_dir, "synthetic", tr, seed, fh=fh)
        else:
            meth, k = T4_SPEC[m]
            nlib.run_single(data, cfg, out_dir, meth, "synthetic", tr, seed, fh=fh, n_nets=k)
    else:
        raise ValueError("unknown job " + j)
    open(done_flag, "w").write("ok")
    nlib.log(f"job {j} finished", fh)


if __name__ == "__main__":
    main()
