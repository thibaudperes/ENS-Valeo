"""Run everything from the images to a submission file.

Embeddings are extracted once, then reused: only this step needs the train dependency group.

    uv run python scripts/run_pipeline.py          # never declares GOOD
    uv run python scripts/run_pipeline.py --good   # declares GOOD when very sure
"""

import argparse
from pathlib import Path

import pandas as pd

from valeo_qc.data import CLASS_NAMES, load_test, load_train
from valeo_qc.embedding_cache import CACHE_DIR, cache_path
from valeo_qc.pipeline import predict_test
from valeo_qc.preprocessing import VIEWS
from valeo_qc.submission import SUBMISSIONS_DIR, write_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--good", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not all(cache_path(view, args.cache_dir).exists() for view in VIEWS):
        from extract_embeddings import extract  # torch is only needed here

        extract(args.cache_dir)

    labels = predict_test(load_train(), load_test(), args.cache_dir, use_good=args.good)
    output = args.output or SUBMISSIONS_DIR / f"final_{'good' if args.good else 'no_good'}.csv"
    write_submission(labels, output)
    counts = pd.Series(labels).map(lambda label: CLASS_NAMES[label]).value_counts()
    print(f"{output}: {counts.to_dict()}")


if __name__ == "__main__":
    main()
