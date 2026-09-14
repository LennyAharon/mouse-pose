# paper_figures/

**Plotting code only.** Nothing in this folder trains, converts, or evaluates a model. Each
script reads results that already exist (evaluation CSVs under `results_dir`, label CSVs under
`data_dir`, both resolved from `paths.yaml`) and writes one figure (PDF) or one LaTeX table.

To produce those results, use the pipeline in the repository `README.md` and the few-shot runner
`scripts/fewshot_cell.sh`. None of the scripts here will work until those runs exist on disk.

## Running a script

Every script takes `--out` and prints its own usage with `--help`:

```bash
python paper_figures/make_fig_aug.py --out figures/fig_aug.pdf
```

The usage line in each docstring writes to `../paper/`, the paper checkout that sits beside
this repository on the authors' machine. Point `--out` wherever you like.

The ensemble-standard-deviation plots (pixel error against keypoint difficulty) share their
machinery through `mouse_pose/plots/ensemble.py`.

## What each script makes

Figures included in the paper as produced:

| script | output | shows |
|---|---|---|
| `make_fig_arch.py` | `fig_arch.pdf` | shared vs per-dataset heads at sampling temperature T = 1, 2, ∞ |
| `make_fig_aug.py` | `fig_aug.pdf` | zero-shot transfer: DLC-style vs per-dataset zoom-in/out augmentation |
| `make_fig_aug_examples.py` | `fig_aug_examples.pdf` | what each dataset's augmentation pipeline produces |
| `make_fig_classes.py` | `fig_classes.pdf` | few-shot adaptation split into supported and new keypoints |
| `make_fig_facemap.py` | `fig_facemap.pdf` | Facemap's inherited pupil under each adaptation method |
| `make_fig_qualitative.py` | `fig_qualitative.pdf` | one test frame with a withheld keypoint, per adaptation method |
| `make_fig_transfer.py` | `fig_transfer.pdf` | keypoints each dataset inherits from the all-data model |

Tables included in the paper as produced:

| script | output | shows |
|---|---|---|
| `make_masked_table.py` | `table_masked.tex` | masked-label protocol, every setting and method |
| `make_table_sampling.py` | `table_sampling.tex` | dataset sampling shares at each temperature |
| `make_table_zoom.py` | `table_zoom.tex` | per-dataset zoom-augmentation ranges |
| `make_dataset_tables.py` | `tables_dataset.tex` | corpus composition and keypoint coverage |

Panels and drafts behind the paper's first three figures, which were assembled by hand:

| script | output | related to |
|---|---|---|
| `make_fig_datasets.py` | `fig_datasets.pdf` | Figure 1: one annotated frame per camera view |
| `make_fig_corpus_bars.py` | `fig_corpus_bars.pdf` | Figure 1: corpus size by frames, observations, vocabulary |
| `make_fig_architecture.py` | `fig_architecture.pdf` | Figure 2: training and adaptation overview |
| `make_fig_anchor.py` | `fig_anchor.pdf` | Figure 2: what anchored LoRA does to each output channel |
| `make_fig_results.py` | `fig_results.pdf` | Figure 3: few-shot curves and masked-label summary |

Because Figures 1 to 3 were composed by hand, rerunning these scripts regenerates the panels,
not the final figures.
