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

The score is a penalty-weighted accuracy: `1 - mean penalty / 10000`. The challenge page and the supplied notebook disagree on two kinds of cells. The leaderboard uses the notebook matrix, which is the default in `src/valeo_qc/metrics.py`.

| True → predicted | Page | Notebook |
|---|---:|---:|
| GOOD → Drift | 100 | 10000 |
| Defect → Drift | 1 | 1000 |

Minimising the expected penalty gives this decision rule:

- **GOOD** requires P(GOOD) > 0.99.
- **Drift** requires P(drift) > 0.50 + 4.45 × P(GOOD), so it is never chosen above P(GOOD) ≈ 0.09.
- **Otherwise**, the most likely defect.

## Leaderboard analysis

Metric analysis from two constant submissions, not a model result. They identify the matrix and the composition of the evaluated images. Under the page matrix, no composition gives these scores. Under the notebook matrix, one integer solution exists, with 528 evaluated images, about half of the test set.

| Submission | Public score |
|---|---:|
| All Drift | 0.8089 |
| All Missing | 0.9860 |

| Class | Images | Share | All Missing penalty per image |
|---|---:|---:|---:|
| Drift | 68 | 12.9% | 128.8 |
| GOOD | 61 | 11.6% | 11.6 |
| Missing | 374 | 70.8% | 0 |
| Other defects | 25 | 4.7% | 0.05 |

`uv run python scripts/leaderboard_analysis.py` writes `results/leaderboard_constants.json`. These shares are not used by the decision code. They only set realistic ranges for threshold analysis.

## Preprocessing

Each image is rotated to a common orientation and cropped, using the per-component parameters of the challenge notebook, copied into `src/valeo_qc/alignment.py`. 21 of the 9,333 images are magnified by a constant factor of about 1.85, whatever their size, so they are first scaled by 0.54 and centred. On these images, the median correlation with the same defects at the base scale is:

| Component | Oversized images | Fixed crop | Resize to base size | Scale and centre |
|---|---:|---:|---:|---:|
| Die02 | 12 | -0.129 | 0.377 | 0.616 |
| Die03 | 7 | 0.029 | 0.583 | 0.668 |
| Die04 | 2 | 0.082 | 0.227 | 0.227 |

Die04 is not solved by this rule. `uv run python scripts/alignment_report.py` reproduces the table.
