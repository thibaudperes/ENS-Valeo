# Valeo visual quality control

Image classification and out-of-domain detection for [ENS Challenge Data n°157](https://challengedata.ens.fr/challenges/157). Work in progress.

## Setup

```bash
uv sync
uv run pre-commit install
uv run pytest
```

Challenge files go in `data/raw/` and are never committed: they are available only to registered participants.

## Metric

The score is a penalty-weighted accuracy: `1 - mean penalty / 10000`. The challenge page and the supplied notebook disagree on two kinds of cells, so both matrices are implemented in `src/valeo_qc/metrics.py`.

| True → predicted | Page | Notebook |
|---|---:|---:|
| GOOD → Drift | 100 | 10000 |
| Defect → Drift | 1 | 1000 |

`uv run python scripts/constant_tables.py` prints the expected score of constant predictions under both matrices.

## Preprocessing

Each image is rotated to a common orientation and cropped, using the per-component parameters of the challenge notebook, copied into `src/valeo_qc/alignment.py`. 21 of the 9,333 images are magnified by a constant factor of about 1.85, whatever their size, so they are first scaled by 0.54 and centred. On these images, the median correlation with the same defects at the base scale is:

| Component | Oversized images | Fixed crop | Resize to base size | Scale and centre |
|---|---:|---:|---:|---:|
| Die02 | 12 | -0.129 | 0.377 | 0.616 |
| Die03 | 7 | 0.029 | 0.583 | 0.668 |
| Die04 | 2 | 0.082 | 0.227 | 0.227 |

Die04 is not solved by this rule. `uv run python scripts/alignment_report.py` reproduces the table.
