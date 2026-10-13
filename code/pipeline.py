"""pipeline.py - schedules all training jobs over the available GPUs and assembles the submission.

Used by the Kaggle notebook and by run_all.py (command-line reproduction).
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES_FALLBACK = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# priority order; a job starts as soon as a GPU is free and its dependencies are finished
JOBS = [
    ("t4_divmix", []),
    ("t4_elrmix", []),
    ("t4_elr", []),
    ("t5_best_synth", ["t4_divmix", "t4_elr", "t4_elrmix"]),
    ("t1_noisy", []),
    ("t1_clean", []),
    ("t5_base_synth", []),
    ("t3_fold0", []),
    ("t3_fold1", []),
    ("t3_fold2", []),
]


def default_cfg(data_dir, out_root, seed=0, epochs=60):
    return {
        "data_dir": data_dir, "out_root": out_root, "seed": seed,
        "epochs": epochs, "batch_size": 128, "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
        "amp": True, "n_folds": 3, "mixup_alpha": 1.0,
        "divmix": {"warmup": 10, "T": 0.5, "alpha": 4.0, "lambda_u": 25.0, "p_threshold": 0.5},
    }


# ----------------------------------------------------------------------------- job scheduler
def run_jobs(cfg, gpus, jobs=None, poll=60):
    """Run jobs in parallel, one subprocess per GPU. `gpus` is a list of GPU ids (e.g. [0, 1]) or [None] for CPU."""
    jobs = jobs or JOBS
    os.makedirs(cfg["out_root"], exist_ok=True)
    cfg_path = os.path.join(cfg["out_root"], "cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    names = [j for j, _ in jobs]
    deps = dict(jobs)
    state = {j: ("done" if os.path.exists(os.path.join(cfg["out_root"], j, "DONE")) else "pending") for j in names}
    running = {}
    lock = threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                for j in names:   # a job whose dependency failed can never run
                    if state[j] == "pending" and any(state[d] == "failed" for d in deps[j]):
                        state[j] = "failed"
                ready = [j for j in names if state[j] == "pending"
                         and all(state[d] == "done" for d in deps[j])]
                if ready:
                    job = ready[0]
                    state[job] = "running"
                    running[gpu] = job
                elif not any(state[j] == "pending" for j in names):
                    return
                else:
                    job = None   # waiting for a dependency that runs on the other GPU
            if job is None:
                time.sleep(10)
                continue
            env = dict(os.environ)
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            os.makedirs(os.path.join(cfg["out_root"], job), exist_ok=True)
            out_log = open(os.path.join(cfg["out_root"], job, "stdout.log"), "a")
            rc = subprocess.call([sys.executable, os.path.join(HERE, "run_job.py"), "--job", job, "--cfg", cfg_path],
                                 stdout=out_log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            out_log.close()
            with lock:
                state[job] = "done" if rc == 0 and os.path.exists(os.path.join(cfg["out_root"], job, "DONE")) else "failed"
                running.pop(gpu, None)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(poll)
        with lock:
            snap = dict(state)
            run = dict(running)
        lines = [f"[{(time.time() - t0) / 60:6.1f} min] " + " ".join(f"{j}:{s}" for j, s in snap.items())]
        for g, j in run.items():
            lp = os.path.join(cfg["out_root"], j, "train.log")
            last = ""
            if os.path.exists(lp):
                with open(lp) as f:
                    ls = f.read().strip().splitlines()
                    last = ls[-1] if ls else ""
            lines.append(f"   gpu {g}: {last}")
        print("\n".join(lines), flush=True)
    failed = [j for j, s in state.items() if s != "done"]
    if failed:
        print("FAILED / NOT RUN:", failed)
        for j in failed:
            p = os.path.join(cfg["out_root"], j, "stdout.log")
            if os.path.exists(p):
                print(f"----- tail of {p}")
                print("".join(open(p).readlines()[-30:]))
    else:
        print(f"All {len(names)} jobs finished in {(time.time() - t0) / 60:.1f} min")
    return state


# ----------------------------------------------------------------------------- helpers
def load_labels(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "labels.csv")).sort_values("index").reset_index(drop=True)
    tr = np.loadtxt(os.path.join(data_dir, "train_idx.txt"), dtype=np.int64).reshape(-1)
    va = np.loadtxt(os.path.join(data_dir, "val_idx.txt"), dtype=np.int64).reshape(-1)
    cls_path = os.path.join(data_dir, "classes.txt")
    classes = [c.strip() for c in open(cls_path) if c.strip()] if os.path.exists(cls_path) else CLASSES_FALLBACK
    return df, tr, va, classes


def rjson(path):
    with open(path) as f:
        return json.load(f)


def write_pred(path, pred):
    pred = np.asarray(pred).astype(int)
    assert pred.shape == (10000,), pred.shape
    pd.DataFrame({"id": np.arange(len(pred)), "pred": pred}).to_csv(path, index=False)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def prec_at_k(y, s, k):
    order = np.argsort(-s, kind="stable")
    return float(y[order[:k]].mean())


def rank01(s):
    from scipy.stats import rankdata
    return (rankdata(s) - 1) / (len(s) - 1)


def words(text):
    return len(text.split())


METHOD_NAMES = {"divmix": "DivideMix", "elr": "ELR (2-net ensemble)", "elrmix": "ELR+mixup (2-net ensemble)"}


# ----------------------------------------------------------------------------- assemble submission
def assemble(cfg, name, roll_no, sub_root, gpu_name="", checker_dir=None):
    out = cfg["out_root"]
    df, tr, va, classes = load_labels(cfg["data_dir"])
    noisy, synth, clean = df.noisy_label.values, df.synthetic_label.values, df.clean_label.values
    pred_dir = os.path.join(sub_root, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    analysis = {}

    # ---------------- Task 1: noise rate + confusion matrix (clean labels used for MEASUREMENT only)
    noise_rate = float((noisy[tr] != clean[tr]).mean())
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (clean[tr], noisy[tr]), 1)
    cm_df = pd.DataFrame(cm, index=[f"clean_{c}" for c in classes], columns=[f"noisy_{c}" for c in classes])
    os.makedirs(os.path.join(out, "analysis"), exist_ok=True)
    cm_df.to_csv(os.path.join(out, "analysis", "t1_confusion_matrix.csv"))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)
    sym = {}
    for i in range(10):
        for j in range(i + 1, 10):
            sym[(i, j)] = off[i, j] + off[j, i]
    sym_pairs = sorted(sym.items(), key=lambda kv: -kv[1])
    n_wrong = int(off.sum())
    animals = {2, 3, 4, 5, 6, 7}   # bird, cat, deer, dog, frog, horse (others are vehicles)
    same_group = sum(off[i, j] for i in range(10) for j in range(10)
                     if (i in animals) == (j in animals)) / max(n_wrong, 1)
    analysis["same_group_share"] = float(same_group)
    analysis["noise_rate_train"] = noise_rate
    analysis["noise_rate_val"] = float((noisy[va] != clean[va]).mean())
    analysis["synthetic_noise_rate_train"] = float((synth[tr] != clean[tr]).mean())

    r_clean = rjson(os.path.join(out, "t1_clean", "result.json"))
    r_noisy = rjson(os.path.join(out, "t1_noisy", "result.json"))
    write_pred(os.path.join(pred_dir, "t1_clean_ref.csv"), np.load(os.path.join(out, "t1_clean", "test_pred_final.npy")))
    write_pred(os.path.join(pred_dir, "t1_noisy_ce.csv"), np.load(os.path.join(out, "t1_noisy", "test_pred_final.npy")))

    # ---------------- Task 2: dynamics, memorization, early stopping
    H = r_noisy["history"]
    dyn = pd.DataFrame({
        "epoch": [h["epoch"] for h in H],
        "train_acc_correct": [round(h["train_acc_correct"], 6) for h in H],
        "train_acc_mislabeled": [round(h["train_acc_mislabeled"], 6) for h in H],
        "val_acc_noisy": [round(h["val_acc_noisy"], 6) for h in H],
        "val_acc_clean": [round(h["val_acc_clean"], 6) for h in H],
    })
    dyn.to_csv(os.path.join(pred_dir, "t2_dynamics.csv"), index=False)
    m = dyn.train_acc_mislabeled.values
    s = pd.Series(m).rolling(3, center=True, min_periods=1).mean().values
    lo = min(3, len(s) - 1)
    ref_ep = lo + int(np.argmin(s[lo:]))                 # 0-based epoch of the plateau minimum
    thr = s[ref_ep] + 0.05
    mem_start = None
    for e in range(ref_ep, len(s)):
        if (s[e:] > thr).all():
            mem_start = e
            break
    if mem_start is None:
        mem_start = int(np.argmax(np.diff(s))) + 1 if len(s) > 1 else 0
    mem_start_epoch = int(dyn.epoch.values[mem_start])
    clean_peak = int(dyn.val_acc_clean.values.argmax())
    es_epoch = int(r_noisy["best"]["epoch"])             # chosen with NOISY val accuracy only
    shutil.copy(os.path.join(out, "t1_noisy", "test_pred_best.npy"), os.path.join(out, "analysis", "t2_es.npy"))
    write_pred(os.path.join(pred_dir, "t2_early_stop.csv"), np.load(os.path.join(out, "t1_noisy", "test_pred_best.npy")))
    analysis.update(mem_ref_epoch=int(dyn.epoch.values[ref_ep]), mem_ref_acc=float(s[ref_ep]),
                    clean_peak_epoch=int(dyn.epoch.values[clean_peak]),
                    clean_peak_acc=float(dyn.val_acc_clean.values[clean_peak]))

    # ---------------- Task 3: noise scores (5 training runs: t1_noisy + 3 CV folds + DivideMix)
    n50 = len(df)
    # (A) cross-validated confidence: 1 - out-of-fold p(noisy label)
    oof = np.full((n50, 10), np.nan, dtype=np.float32)
    for f in range(cfg["n_folds"]):
        d = os.path.join(out, f"t3_fold{f}")
        ids = np.load(os.path.join(d, "heldout_ids.npy"))
        oof[ids] = np.load(os.path.join(d, "heldout_probs_best.npy"))
    assert not np.isnan(oof[tr]).any(), "missing out-of-fold predictions"
    sA = 1.0 - oof[tr, noisy[tr]]
    # (B) area-under-margin of the Task-1 noisy CE run, averaged up to the early-stop epoch (noisy-val chosen)
    marg = np.load(os.path.join(out, "t1_noisy", "train_margins.npy")).astype(np.float32)
    sB = -marg[:es_epoch].mean(0)
    # (C) DivideMix final per-sample loss (two networks)
    dm_ids = np.load(os.path.join(out, "t4_divmix", "train_ids.npy"))
    assert (dm_ids == tr).all()
    sC = np.load(os.path.join(out, "t4_divmix", "train_loss_score.npy")).astype(np.float64)
    score = (rank01(sA) + rank01(sB) + rank01(sC)) / 3.0      # fixed a-priori ensemble (no clean-label tuning)
    pd.DataFrame({"index": tr, "score": np.round(score, 8)}).to_csv(os.path.join(pred_dir, "t3_noise_scores.csv"), index=False)
    # self-check with clean labels (measurement only - does not change the submitted score)
    ytrue = (noisy[tr] != clean[tr]).astype(int)
    k = int(ytrue.sum())
    t3 = {}
    for nm, sc in [("cv_confidence", sA), ("aum_early", sB), ("divmix_loss", sC), ("ensemble", score)]:
        t3[nm] = {"auroc": auroc(ytrue, sc), "prec_at_k": prec_at_k(ytrue, sc, k)}
    analysis["task3_check"] = t3
    analysis["task3_k"] = k
    # failure mode: mislabeled images that the ensemble ranks below the top-k
    order = np.argsort(-score, kind="stable")
    in_top = np.zeros(len(tr), bool)
    in_top[order[:k]] = True
    missed = (ytrue == 1) & ~in_top
    agree = (oof[tr].argmax(1) == noisy[tr])
    analysis["t3_missed"] = int(missed.sum())
    analysis["t3_missed_model_agrees_wrong"] = float(agree[missed].mean()) if missed.any() else 0.0
    mp = {}
    for c_, n_ in zip(clean[tr][missed], noisy[tr][missed]):
        mp[(c_, n_)] = mp.get((c_, n_), 0) + 1
    top_miss = sorted(mp.items(), key=lambda kv: -kv[1])[:2]
    analysis["t3_top_missed_pairs"] = [(classes[a], classes[b], v) for (a, b), v in top_miss]

    # ---------------- Task 4
    t4 = []
    for mth in ["divmix", "elr", "elrmix"]:
        d = os.path.join(out, f"t4_{mth}")
        r = rjson(os.path.join(d, "result.json"))
        fname = f"t4_{mth}.csv"
        write_pred(os.path.join(pred_dir, fname), np.load(os.path.join(d, "test_pred_best.npy")))
        t4.append({"name": mth, "file": fname, "val_acc_noisy_labels": round(r["best"]["val_acc_noisy"], 6),
                   "_epoch": r["best"]["epoch"], "_val_clean": r["best"]["val_acc_clean"],
                   "_final_val_noisy": r["final"]["val_acc_noisy"]})
    best = max(t4, key=lambda x: x["val_acc_noisy_labels"])["name"]

    # ---------------- Task 5
    r5b = rjson(os.path.join(out, "t5_base_synth", "result.json"))
    r5m = rjson(os.path.join(out, "t5_best_synth", "result.json"))
    m5 = open(os.path.join(out, "t5_best_synth", "method.txt")).read().strip()
    assert m5 == best, f"t5_best_synth used '{m5}' but best Task-4 method is '{best}'"
    write_pred(os.path.join(pred_dir, "t5_baseline_synth.csv"), np.load(os.path.join(out, "t5_base_synth", "test_pred_final.npy")))
    write_pred(os.path.join(pred_dir, "t5_best_synth.csv"), np.load(os.path.join(out, "t5_best_synth", "test_pred_best.npy")))

    # ---------------- results.json
    results = {
        "name": name, "roll_no": roll_no, "seed": int(cfg["seed"]),
        "training_setup": {
            "optimizer": f"SGD (momentum {cfg['momentum']}, weight decay {cfg['weight_decay']}), lr {cfg['lr']}",
            "learning_rate_schedule": f"cosine annealing per epoch from {cfg['lr']} to ~0 over {cfg['epochs']} epochs (no restarts)",
            "batch_size": int(cfg["batch_size"]), "epochs": int(cfg["epochs"]),
            "augmentation": "random crop 32x32 with 4-px zero padding + random horizontal flip (val/test predictions average the image and its horizontal flip)",
            "gpu_used": gpu_name or "GPU",
        },
        "task1": {
            "noise_rate": round(noise_rate, 6),
            "val_acc_noisy_labels__clean_ref_model": round(r_clean["final"]["val_acc_noisy"], 6),
            "val_acc_noisy_labels__noisy_ce_model": round(r_noisy["final"]["val_acc_noisy"], 6),
        },
        "task2": {
            "memorization_start_epoch": mem_start_epoch,
            "early_stop_epoch": es_epoch,
            "val_acc_noisy_labels_at_early_stop": round(r_noisy["best"]["val_acc_noisy"], 6),
        },
        "task3": {"method": "rank-average ensemble of 3 signals: (a) 3-fold cross-validated confidence 1-p(noisy label), "
                            "(b) negative area-under-margin of the Task-1 CE run up to the early-stop epoch, "
                            "(c) DivideMix per-sample loss (2 nets) - 5 training runs total"},
        "task4": {"methods": [{k_: v for k_, v in x.items() if not k_.startswith("_")} for x in t4], "best_method": best},
        "task5": {
            "val_acc_synthetic_labels__baseline": round(r5b["final"]["val_acc_synthetic"], 6),
            "val_acc_synthetic_labels__best_method": round(r5m["best"]["val_acc_synthetic"], 6),
        },
    }
    with open(os.path.join(sub_root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- report.md
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    top3 = sym_pairs[:3]
    share3 = sum(v for _, v in top3) / max(n_wrong, 1)
    q1 = (f"Noise rate is {pct(noise_rate)} ({n_wrong} of {len(tr)} training labels). Three class pairs, "
          + ", ".join(f"{classes[i]}<->{classes[j]} ({v})" for (i, j), v in top3)
          + f", cause {pct(share3)} of all wrong labels. The biggest single directions are "
          + ", ".join(f"{classes[i]}->{classes[j]} ({v})" for v, i, j in pairs[:3])
          + f". The noise is structured, not uniform: {pct(same_group)} of wrong labels stay within the same "
            "super-group (animal->animal or vehicle->vehicle).")
    q2 = (f"Memorization starts around epoch {mem_start_epoch}. Training accuracy on the mislabeled subset (fit to the wrong label) "
          f"bottoms out near {pct(analysis['mem_ref_acc'])} at epoch {analysis['mem_ref_epoch']}; from epoch {mem_start_epoch} it stays "
          f"5+ points higher and climbs to {pct(m[-1])} by epoch {int(dyn.epoch.values[-1])}, "
          f"while correct-subset accuracy reaches {pct(dyn.train_acc_correct.values[-1])}. Clean-label validation accuracy peaks at "
          f"epoch {analysis['clean_peak_epoch']} ({pct(analysis['clean_peak_acc'])}) and falls to {pct(dyn.val_acc_clean.values[-1])} at the end. "
          f"Noisy-label validation accuracy peaks at epoch {es_epoch}, which we use for early stopping.")
    sig_names = {"cv_confidence": "cross-validated confidence", "aum_early": "early area-under-margin",
                 "divmix_loss": "DivideMix loss", "ensemble": "rank ensemble"}
    best_sig = max(["cv_confidence", "aum_early", "divmix_loss"], key=lambda z: t3[z]["auroc"])
    miss_pairs = ", ".join(f"{a}->{b}" for a, b, _ in analysis["t3_top_missed_pairs"])
    q3 = (f"On our own clean-label check, the best single signal was {sig_names[best_sig]} (AUROC {t3[best_sig]['auroc']:.3f}); "
          f"the submitted rank ensemble scores AUROC {t3['ensemble']['auroc']:.3f}, precision@k {t3['ensemble']['prec_at_k']:.3f}. "
          f"Failure mode: systematic human confusions. For {pct(analysis['t3_missed_model_agrees_wrong'])} of missed wrong labels "
          f"the held-out model itself predicts the wrong label, because the image genuinely looks like that class "
          f"(most missed: {miss_pairs}), so its loss stays low.")
    t4d = {x["name"]: x for x in t4}
    b = t4d[best]
    ce_es, ce_fin = r_noisy["best"], r_noisy["final"]
    why = {
        "divmix": ("A GMM on per-sample loss splits data into likely-clean (kept) and likely-noisy (labels dropped, used as "
                   "unlabeled data); each network trains on the other's split, avoiding self-confirmation, and MixMatch "
                   "with co-guessed targets still learns from noisy images instead of memorizing them."),
        "elr": ("ELR pulls predictions towards a running average of the model's own early predictions, so gradients from "
                "wrong labels are cancelled once the early-learning phase has captured the clean signal, slowing memorization."),
        "elrmix": ("ELR keeps predictions close to the model's own early (mostly correct) predictions, cancelling gradients from "
                   "wrong labels; mixup blends images and labels, so no single wrong label can be memorized exactly; "
                   "averaging two networks reduces variance."),
    }[best]
    others = ", ".join(f"{x['name']} {pct(x['val_acc_noisy_labels'])}" for x in t4 if x["name"] != best)
    bv = b["val_acc_noisy_labels"]
    if bv > ce_es["val_acc_noisy"]:
        verdict = "It helped, beating both final and early-stopped CE."
    elif bv > ce_fin["val_acc_noisy"]:
        verdict = "It beat final CE but not early-stopped CE, so it mainly delayed memorization."
    else:
        verdict = "It did not beat CE within the 60-epoch budget, which leaves little training time to exploit the mechanism below."
    q4 = (f"Best was {METHOD_NAMES[best]}: noisy-val {pct(bv)} (clean-val {pct(b['_val_clean'])}) vs CE "
          f"{pct(ce_fin['val_acc_noisy'])} final / {pct(ce_es['val_acc_noisy'])} early-stopped; others: {others}. "
          f"{verdict} {why}")
    d_real = b["_val_clean"] - ce_fin["val_acc_clean"]
    d_syn = r5m["best"]["val_acc_clean"] - r5b["final"]["val_acc_clean"]
    same = (d_real > 0) == (d_syn > 0)
    bigger = "larger" if abs(d_syn) > abs(d_real) else "smaller"
    q5 = (f"Measured by clean-label validation accuracy, {best} vs CE baseline is {d_real * 100:+.1f} points on human noise and "
          f"{d_syn * 100:+.1f} on synthetic noise (CE baseline: {pct(ce_fin['val_acc_clean'])} real, {pct(r5b['final']['val_acc_clean'])} synthetic). "
          + ("The ranking did not change" if same else "The ranking changed")
          + f"; the gap is {bigger} on synthetic noise. Symmetric noise is random and feature-independent, so wrong labels "
            "conflict with the image, have high loss and are easy to separate. Human noise sits in ambiguous look-alike pairs, "
            "so wrong labels look plausible, are learned like signal and are harder to filter.")
    report = "# Report - Learning from Messy Human Labels (CIFAR-10N)\n\n" + "\n\n".join(
        f"Q{i}. {t}" for i, t in enumerate([q1, q2, q3, q4, q5], 1)) + "\n"
    for i, t in enumerate([q1, q2, q3, q4, q5], 1):
        if words(t) > 100:
            print(f"WARNING: Q{i} has {words(t)} words (>100) - please shorten it manually")
    with open(os.path.join(sub_root, "report.md"), "w") as f:
        f.write(report)

    # ---------------- code/
    code_dir = os.path.join(sub_root, "code")
    os.makedirs(code_dir, exist_ok=True)
    for fn in ["nlib.py", "run_job.py", "pipeline.py", "run_all.py", "model.py", "README.md"]:
        src = os.path.join(HERE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(code_dir, fn))
    for nb in [p for p in os.listdir(HERE) if p.endswith(".ipynb")]:
        shutil.copy(os.path.join(HERE, nb), os.path.join(code_dir, nb))

    analysis.update(t4=t4, best=best, results=results, d_real=d_real, d_syn=d_syn,
                    confusion=cm.tolist(), word_counts=[words(t) for t in [q1, q2, q3, q4, q5]])
    with open(os.path.join(out, "analysis", "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=1, default=str)
    return results, analysis, report


def make_zip(sub_root, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(sub_root):
            for fn in files:
                if "__pycache__" in d:
                    continue
                full = os.path.join(d, fn)
                z.write(full, os.path.relpath(full, sub_root))
    return zip_path


def run_checker(check_format_py, data_dir, target, work):
    """check_format.py looks for data/train_idx.txt next to itself -> run a copy with a data/ symlink."""
    cdir = os.path.join(work, "checker")
    os.makedirs(os.path.join(cdir, "data"), exist_ok=True)
    shutil.copy(check_format_py, os.path.join(cdir, "check_format.py"))
    shutil.copy(os.path.join(data_dir, "train_idx.txt"), os.path.join(cdir, "data", "train_idx.txt"))
    r = subprocess.run([sys.executable, os.path.join(cdir, "check_format.py"), target],
                       capture_output=True, text=True, cwd=cdir)
    print(r.stdout + r.stderr)
    return "All good." in r.stdout
