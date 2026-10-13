"""Reproduce every task end to end from the command line (same code the Kaggle notebook runs).

  python run_all.py --data_dir /path/to/data --model_py /path/to/model.py \
                    --name "Your Name" --roll_no 12345678 --out ./work [--gpus 0,1] [--check_format /path/to/check_format.py]
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_py", default=os.path.join(HERE, "model.py"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--roll_no", required=True)
    ap.add_argument("--out", default="./work")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gpus", default="auto", help="comma-separated GPU ids, 'auto', or 'cpu'")
    ap.add_argument("--check_format", default=None)
    a = ap.parse_args()

    if os.path.abspath(a.model_py) != os.path.join(HERE, "model.py"):
        shutil.copy(a.model_py, os.path.join(HERE, "model.py"))
    if a.gpus == "auto":
        import torch
        gpus = list(range(torch.cuda.device_count())) or [None]
    elif a.gpus == "cpu":
        gpus = [None]
    else:
        gpus = [int(g) for g in a.gpus.split(",")]
    gpu_name = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}"
    except Exception:  # noqa: BLE001
        pass

    out = os.path.abspath(a.out)
    cfg = pipeline.default_cfg(os.path.abspath(a.data_dir), os.path.join(out, "runs"), seed=a.seed, epochs=a.epochs)
    state = pipeline.run_jobs(cfg, gpus)
    if any(s != "done" for s in state.values()):
        sys.exit("some jobs failed - see the logs above")
    sub = os.path.join(out, "submission")
    pipeline.assemble(cfg, a.name, a.roll_no, sub, gpu_name=gpu_name)
    zp = pipeline.make_zip(sub, os.path.join(out, f"{a.roll_no}_cv_bootcamp.zip"))
    print("wrote", zp)
    if a.check_format:
        pipeline.run_checker(a.check_format, cfg["data_dir"], zp, out)


if __name__ == "__main__":
    main()
