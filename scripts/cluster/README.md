# Running Mighty Mouse experiments on a SLURM cluster

Sketch of the pipeline for a shared GPU cluster (target: Georgia Tech, B200 nodes, SLURM).
Everything here already runs locally on one GPU; the cluster version changes only *who launches
the cells*, not what a cell is.

## Shape of the workload

An experiment round is a grid of **independent single-GPU training jobs**. Each job trains one
pose model on one training CSV with one seed, then evaluates it on every dataset's test set.
There is no communication between jobs and no multi-GPU training. That maps onto SLURM as one
job array per round.

| what | count per round | notes |
|---|---|---|
| all-data trunk (Mighty Mouse) | 1 per seed | the model of record |
| dedicated single-dataset models | 6 per seed | baselines |
| leave-one-out trunks | 6 per seed | zero-shot transfer to an unseen lab |
| ablations (schedule, augmentation, ...) | 1 to 7 per variant | same shape, different config |
| few-shot adaptation cells | ~180, short | next stage, same pattern (not in this sketch) |

Three seeds of the recipe grid = 39 jobs. Rounds repeat every time the labeled corpus changes.

## Per-job resource profile (measured on an NVIDIA L4, 24 GB)

| resource | per job |
|---|---|
| GPU | 1, ~6 GB memory (ViT-S DINOv3, batch 32, 256x256 inputs) |
| CPU | 8 cores: image augmentation is CPU-bound and is the bottleneck; 2 jobs saturate 8 cores |
| RAM | ~6 GB |
| wall time | ~2.5 h for 12,000 steps on an L4 alone; 24,000-step schedule under test |
| output | ~260 MB per job with checkpoint (103 MB checkpoint + predictions/metrics CSVs) |

Expectation on a B200: GPU time per job well under an hour; the CPU augmentation pipeline then
limits throughput unless more cores per job are available or augmentation moves to the GPU
(Lightning Pose supports NVIDIA DALI for video; labeled-frame augmentation is imgaug on CPU).
This is the first thing worth discussing with the cluster engineers.

## Files

- `fit_one.py` — fits and evaluates ONE model from command-line arguments. Idempotent (skips a
  cell that already has results, retrains a cell that died). Non-zero exit on failure.
- `fit_one.slurm` — the sbatch template: one array task = one call of `fit_one.py`, reading its
  cell from a tab-separated jobs file by `$SLURM_ARRAY_TASK_ID`. Resource lines = the profile above.
- `orchestrate.py` — enumerates the cells of a round (`all`, `dedicated`, `loo`, `ablation:<name>`,
  x seeds), drops cells that already have results, writes the jobs file and submits one job array
  with a concurrency throttle. `status` lists queue and finished cells; `collect` builds the
  evaluation tables (`scripts/eval_suite.py`) from everything finished.

```
python scripts/cluster/orchestrate.py plan   all dedicated loo --seeds "0;1;2"   # 39 cells
python scripts/cluster/orchestrate.py submit all dedicated loo --seeds "0;1;2" --max_concurrent 16
python scripts/cluster/orchestrate.py status
python scripts/cluster/orchestrate.py collect
```

## Data and storage

- Inputs, read-only, shared by all jobs: labeled frames (~1.3 GB for six datasets) plus label
  CSVs; ~1 GB of videos for qualitative evaluation; raw sources ~3 GB. Frames never change
  between corpus versions; each version is label CSVs plus symlinks.
- Outputs: one directory per job (`<results>/<area>/<tag>_train/.../seed<k>/`), ~10 GB per
  full three-seed grid with checkpoints. Evaluation CSVs are kept permanently; checkpoints are
  pruned after the round unless needed for adaptation experiments.
- Paths are set in one gitignored file (`paths.yaml`: `data_dir`, `results_dir`, `raw_dir`), so a
  cluster mount needs no code change.

## Software

Python 3.12, PyTorch 2.8 + CUDA 12.8, Lightning 2.5, `lightning-pose` (paninski-lab, branch
`post_sub_lp`) and `mouse-pose` (this repo, branch `post_sub_mm`) installed editable from git.
A conda environment file or a container recipe can be provided; the DINOv3 backbone weights are
downloaded once and cached (`~/.cache`), so the cache directory should be on shared storage.

## Open questions for the cluster engineers

1. CPU cores per GPU job: can we get 16 to 32 cores per B200 job, or should augmentation move to
   the GPU? This decides the throughput.
2. Job arrays with a concurrency cap vs. a queue-based dispatcher: is `sbatch --array` with `%N`
   the preferred pattern, and what array size limit applies?
3. Shared filesystem for results: bandwidth for many jobs writing checkpoints and TensorBoard
   logs concurrently (locally a FUSE mount made this slow; scratch + copy-back may be better).
4. Model cache and container: where to keep DINOv3 weights and whether a Singularity/Apptainer
   image is preferred over conda.
5. Whether a small GPU partition exists for the many short few-shot jobs (~20 min each).
