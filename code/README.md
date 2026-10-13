# CIFAR-10N noisy-label bootcamp: how to reproduce

Every network is the ResNet-18 in `model.py`, trained from scratch on the provided files only.
No pretrained weights or external data are used. The seed is 0.

## Files
| file | purpose |
|---|---|
| `nlib.py` | data loading (all images kept on the GPU, GPU augmentation), CE/GCE/ELR single-network training, DivideMix (2 networks) |
| `run_job.py` | runs one training job, e.g. `python run_job.py --job t1_noisy --cfg cfg.json` |
| `pipeline.py` | job scheduler (one job per GPU, in parallel) and assembly of `predictions/`, `results.json`, `report.md` |
| `run_all.py` | command-line entry point for the whole pipeline |
| `*.ipynb` | the Kaggle notebook, if included (it runs exactly the same code) |

## Shared training recipe (baseline and all methods)
SGD with momentum 0.9 and weight decay 5e-4. Learning rate 0.1 with cosine annealing per epoch over 60 epochs. Batch size 128.
Augmentation is a random 32x32 crop with 4-px zero padding plus a horizontal flip. Training uses fp16 mixed precision.
Each network trains for 60 epochs.

## Reproduce everything (one command)
```bash
python run_all.py --data_dir <folder with labels.csv, *.npy, *_idx.txt> --model_py <provided model.py> \
                  --name "<name>" --roll_no <roll> --out ./work --gpus 0,1 --check_format <provided check_format.py>
```
On Kaggle you can instead open the notebook and run all cells (Save Version, then Save & Run All).

## Reproduce each task (individual jobs)
First write a config: `python -c "import pipeline,json; json.dump(pipeline.default_cfg('<data_dir>','./work/runs'), open('cfg.json','w'))"`.
Then run the jobs below, each with `python run_job.py --job <job> --cfg cfg.json`. After that, `pipeline.assemble(...)` writes the files.

| task | job(s) | output file(s) |
|---|---|---|
| Task 1 | `t1_clean` (CE on clean labels, reference), `t1_noisy` (CE on noisy labels) | `t1_clean_ref.csv`, `t1_noisy_ce.csv` (final epoch 60) |
| Task 2 | `t1_noisy` (logs the dynamics every epoch) | `t2_dynamics.csv`, `t2_early_stop.csv` (epoch with the best **noisy** val acc) |
| Task 3 | `t3_fold0..2` (3-fold CV), `t1_noisy`, `t4_divmix`: 5 training runs | `t3_noise_scores.csv` |
| Task 4 | `t4_divmix`, `t4_elr`, `t4_gce` | `t4_divmix.csv`, `t4_elr.csv`, `t4_gce.csv` (checkpoint with the best noisy val acc) |
| Task 5 | `t5_base_synth` (CE, final epoch), `t5_best_synth` (best Task-4 method, checkpoint chosen on synthetic val) | `t5_baseline_synth.csv`, `t5_best_synth.csv` |

## Task 3 score
The score averages the ranks of three signals; the weights were fixed in advance and never tuned on clean labels.
1. Cross-validated confidence: 1 - p(noisy label) from a model that never saw the image (3 folds; each fold's checkpoint is chosen on noisy val).
2. Negative area-under-margin (logit of the given label minus the largest other logit) from the Task-1 CE run, averaged over epochs 1 to the early-stop epoch.
3. Normalised per-sample CE loss under the two final DivideMix networks.

## Rules compliance
- Clean labels are used only for: training `t1_clean`; the Task-1 noise rate and confusion matrix; the Task-2 curves; and our own Task-3 check (AUROC is printed but never used to choose anything).
- Model and checkpoint selection uses noisy val labels in Tasks 1-4 and synthetic val labels in Task 5.
- Budget: at most 60 epochs per network and at most 2 networks per method (DivideMix uses 2). Task 3 uses 5 training runs.
