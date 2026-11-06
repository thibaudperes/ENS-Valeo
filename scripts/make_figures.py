"""Draw the README figures from results/*.json and the embedding caches.

No figure contains a challenge image: only aggregates, 2D projections and penalty tables.
Figures that need the caches (embedding map, drift detector) recompute the detector scores and
stop if their AUROC differs from results/drift.json. Writes reports/figures/*.png.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from valeo_qc.data import CLASS_NAMES, LABEL_IDS, load_train  # noqa: E402
from valeo_qc.drift import out_of_fold_scores, scores_by_lib, thresholds_by_lib  # noqa: E402
from valeo_qc.embedding_cache import cache_path, load_embeddings  # noqa: E402
from valeo_qc.metrics import DRIFT, GOOD, PENALTY_MATRICES  # noqa: E402
from valeo_qc.validation import make_folds  # noqa: E402
from valeo_qc.viz import PALETTE, save_fig, set_conclusion, set_style  # noqa: E402

RESULTS = Path("results")
RARE_DEFECTS = ("Boucle plate", "Lift-off blanc", "Lift-off noir", "Short circuit MOS")
PSEUDO_DRIFT = (*RARE_DEFECTS, "blur")
FALSE_ALARM_RATE = 0.005
AUROC_TOLERANCE = {"knn": 1e-9, "mahalanobis": 5e-3}  # Mahalanobis: randomised PCA solver


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def color_of(name: str) -> str:
    return PALETTE["Drift"] if name == "blur" else PALETTE[name]


def label_of(name: str) -> str:
    return "Blur (synthetic)" if name == "blur" else f"{name} (held out)"


def penalty(score: float) -> float:
    return (1 - score) * PENALTY_MATRICES["notebook"].max()


# a. Score ladder -----------------------------------------------------------------------------


def score_ladder() -> None:
    constants, board = load("leaderboard_constants"), load("public_leaderboard")
    pipeline = constants["pipeline"]["scores"]
    rows = [
        ("All Drift (probe)", constants["scores"]["all_drift"], "neutral"),
        ("Notebook benchmark*", constants["benchmark_pwa_notebook"], "neutral"),
        ("All Missing (probe)", constants["scores"]["all_missing"], "neutral"),
        ("Pipeline, GOOD when very sure", pipeline["good"], "alternative"),
        ("Pipeline, never GOOD (final)", pipeline["no_good"], "final"),
    ]
    assert abs(pipeline["no_good"] - board["final_score"]) < 5e-5
    left = 0.970
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for y, (_, score, role) in enumerate(rows):
        shown = max(score, left)
        ax.plot([left, shown], [y, y], color=PALETTE[role], lw=2.5, solid_capstyle="butt")
        if score < left:
            ax.annotate(
                f"{score:.4f}, off scale",
                (left, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=PALETTE["ink_secondary"],
            )
            continue
        ax.plot(shown, y, "o", ms=9, color=PALETTE[role], mec="white", mew=2)
        if role == "final":
            # above the point, ending at it: nothing to its right, so first place never crosses it
            ax.annotate(
                f"{score:.4f} · {_ordinal(board['rank'])} on the public leaderboard",
                (shown, y),
                xytext=(4, 10),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontweight="bold",
            )
            continue
        ax.annotate(
            f"{score:.4f}",
            (shown, y),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
        )
    first = board["first_place_score"]
    ax.axvline(first, color=PALETTE["ink_secondary"], ls=(0, (4, 3)), lw=1.2)
    ax.annotate(
        f"1st place\n{first:.4f}",
        (first, 0),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=PALETTE["ink_secondary"],
    )
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows])
    ax.set_xlim(left, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Public score (penalty-weighted accuracy, higher is better)")
    ax.set_title("*computed on all 1,055 test images, the others on the 528 public ones")
    set_conclusion(
        fig,
        f"My pipeline scores {pipeline['no_good']:.4f}, {_ordinal(board['rank'])} on the "
        f"public leaderboard, {first - pipeline['no_good']:.4f} from first place",
    )
    save_fig(fig, "a_score_ladder")


def _ordinal(rank: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")


# b. Metric detective -------------------------------------------------------------------------


def metric_matrices() -> None:
    constants = load("leaderboard_constants")
    composition = constants["solutions_by_matrix"]["notebook"][0]
    n = composition["n_images"]
    defects = n - composition["good"] - composition["drift"]
    probe = constants["scores"]["all_drift"]

    def all_drift_score(matrix: np.ndarray) -> float:
        total = composition["good"] * matrix[GOOD, DRIFT] + defects * matrix[1, DRIFT]
        return 1 - total / n / matrix.max()

    labels = list(CLASS_NAMES)
    page, notebook = PENALTY_MATRICES["page"], PENALTY_MATRICES["notebook"]
    differs = page != notebook
    norm = LogNorm(vmin=1, vmax=10000)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(PALETTE["neutral_light"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, (name, matrix) in zip(axes, (("page", page), ("notebook", notebook)), strict=True):
        shown = np.where(matrix == 0, np.nan, matrix).astype(float)
        ax.imshow(shown, cmap=cmap, norm=norm)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]
                dark = value >= 1000
                ax.text(
                    j,
                    i,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if dark else PALETTE["ink"],
                    fontweight="bold" if differs[i, j] else "normal",
                )
                if differs[i, j]:
                    ax.add_patch(
                        Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=PALETTE["Drift"], lw=2.2)
                    )
        score = all_drift_score(matrix)
        ax.set_title(
            f"{'Challenge page' if name == 'page' else 'Supplied notebook'}: "
            f"All Drift would score {score:.4f}"
        )
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted label")
        ax.grid(False)
    axes[0].set_ylabel("True label")
    fig.text(
        0.0,
        -0.04,
        f"Orange boxes: the {differs.sum()} cells where the two matrices differ. "
        f"The All Drift probe submission scored {probe:.4f}; scores use the test composition "
        f"recovered from the two probes ({n} images, {composition['drift']} drift).",
        ha="left",
        color=PALETTE["ink_secondary"],
        fontsize=9.5,
    )
    winner = min(
        ("page", page), ("notebook", notebook), key=lambda m: abs(all_drift_score(m[1]) - probe)
    )
    set_conclusion(
        fig,
        f"One probe submission (All Drift, {probe:.4f}) proves the leaderboard uses the "
        f"{winner[0]} matrix",
    )
    save_fig(fig, "b_metric_matrices")


# c. Where the penalty comes from -------------------------------------------------------------


def penalty_sources() -> None:
    parts = load("leaderboard_constants")["all_missing_penalty_per_image"]
    rows = [
        ("Drift predicted as Missing", parts["undetected_drift"], PALETTE["Drift"]),
        ("GOOD predicted as Missing", parts["good_as_defect"], PALETTE["GOOD"]),
        ("Other defect predicted as Missing", parts["other_defects"], PALETTE["neutral"]),
    ]
    total = sum(value for _, value, _ in rows)
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for y, (_, value, color) in enumerate(rows):
        ax.barh(y, value, color=color, height=0.6)
        ax.annotate(
            f"{value:.1f} points ({value / total:.0%})" if value >= 0.1 else f"{value:.2f} points",
            (value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Penalty per image of the constant All Missing prediction (public test)")
    ax.set_title(f"Total: {total:.1f} points per image, score {1 - total / 10000:.4f}")
    share = parts["undetected_drift"] / total
    set_conclusion(
        fig, f"Undetected drift makes {share:.0%} of the penalty: the detector came first"
    )
    save_fig(fig, "c_penalty_sources")


# Detector scores, recomputed from the caches --------------------------------------------------


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def load_crop_views():
    train = load_train()
    with np.load(cache_path("blur_crop")) as cache:
        blurred = train.set_index("filename").loc[cache["filenames"]].reset_index()
    crop = unit_rows(load_embeddings(cache_path("crop"), train["filename"]))
    blur_crop = unit_rows(load_embeddings(cache_path("blur_crop"), blurred["filename"]))
    return train, blurred, crop, blur_crop


def pseudo_drift_scores(method, train, blurred, features, blur_features, folds) -> dict:
    """Known and pseudo-drift scores, as in scripts/evaluate_drift.py."""
    libs, labels = train["lib"].to_numpy(), train["label"].to_numpy()
    scores = {}
    for defect in RARE_DEFECTS:
        held = labels == LABEL_IDS[defect]
        known = out_of_fold_scores(method, features[~held], libs[~held], folds[~held])
        drift = scores_by_lib(method, features[~held], libs[~held], features[held], libs[held])
        scores[defect] = (known, libs[~held], drift, libs[held])

    known = out_of_fold_scores(method, features, libs, folds)
    blur_libs = blurred["lib"].to_numpy()
    blur_folds = pd.Series(folds, index=train["filename"]).loc[blurred["filename"]].to_numpy()
    drift = np.empty(len(blurred))
    for fold in np.unique(blur_folds):
        sel = blur_folds == fold
        drift[sel] = scores_by_lib(
            method, features[folds != fold], libs[folds != fold], blur_features[sel], blur_libs[sel]
        )
    scores["blur"] = (known, libs, drift, blur_libs)
    return scores


def check_auroc(method: str, scores: dict, reference: dict) -> dict[str, float]:
    """Stop if a recomputed AUROC moves; return the published values."""
    aurocs = {}
    for name, (known, _, drift, _) in scores.items():
        auroc = roc_auc_score(np.r_[np.zeros(len(known)), np.ones(len(drift))], np.r_[known, drift])
        expected = reference[name]["auroc"]
        if abs(auroc - expected) > AUROC_TOLERANCE[method]:
            raise RuntimeError(f"{name}: AUROC {auroc:.4f} differs from results ({expected:.4f})")
        aurocs[name] = expected
    return aurocs


# d. Embedding map ----------------------------------------------------------------------------


def embedding_map(train, blurred, crop, blur_crop) -> None:
    detected = load("drift")["detectors"]["knn_crop"]["blur"][f"at_{FALSE_ALARM_RATE}"]["recall"]
    points = PCA(n_components=50, random_state=0).fit_transform(np.r_[crop, blur_crop])
    xy = TSNE(n_components=2, init="pca", random_state=0, perplexity=40).fit_transform(points)
    known, blur = xy[: len(crop)], xy[len(crop) :]
    labels = train["label"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 7.2))
    missing = labels == LABEL_IDS["Missing"]
    ax.scatter(
        *known[missing].T,
        s=3,
        color=PALETTE["Missing"],
        alpha=0.35,
        lw=0,
        label=f"Missing ({missing.sum():,})",
    )
    for name in ("GOOD", *RARE_DEFECTS):
        sel = labels == LABEL_IDS[name]
        ax.scatter(*known[sel].T, s=7, color=PALETTE[name], lw=0, label=f"{name} ({sel.sum():,})")
    ax.scatter(
        *blur.T,
        s=22,
        marker="x",
        color=PALETTE["Drift"],
        lw=1.1,
        label=f"Blurred copy, pseudo-drift ({len(blur)})",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=2.5)
    ax.set_title(
        f"t-SNE of frozen DINOv2 crop embeddings, {len(crop):,} train images and "
        f"{len(blur)} blurred copies (no image shown)"
    )
    set_conclusion(
        fig,
        f"Blurred copies form clusters of their own; the kNN detector flags {detected:.0%} of them",
    )
    save_fig(fig, "d_embedding_map")


# e. Drift detector ---------------------------------------------------------------------------


def drift_detector(train, blurred, crop, blur_crop) -> None:
    reference = load("drift")["detectors"]
    folds = make_folds(train)
    knn = pseudo_drift_scores("knn", train, blurred, crop, blur_crop, folds)
    maha = pseudo_drift_scores("mahalanobis", train, blurred, crop, blur_crop, folds)
    knn_auroc = check_auroc("knn", knn, reference["knn_crop"])
    maha_auroc = check_auroc("mahalanobis", maha, reference["mahalanobis_crop"])

    fig = plt.figure(figsize=(12, 7.6))
    grid = fig.add_gridspec(2, len(PSEUDO_DRIFT), height_ratios=(1.25, 1))
    ax = fig.add_subplot(grid[0, :])
    grid_x = np.geomspace(0.1, 4.0, 400)
    known, known_libs = knn["blur"][0], knn["blur"][1]
    thresholds = thresholds_by_lib(known, known_libs, FALSE_ALARM_RATE)
    known_ratio = known / np.array([thresholds[lib] for lib in known_libs])
    ax.plot(
        grid_x,
        [(known_ratio > x).mean() for x in grid_x],
        color=PALETTE["neutral"],
        lw=3,
        label=f"Known images, out of fold ({len(known):,}): {(known_ratio > 1).mean():.1%} flagged",
    )
    for name in PSEUDO_DRIFT:
        k, k_libs, drift, d_libs = knn[name]
        th = thresholds_by_lib(k, k_libs, FALSE_ALARM_RATE)
        ratio = drift / np.array([th[lib] for lib in d_libs])
        ax.plot(
            grid_x,
            [(ratio > x).mean() for x in grid_x],
            lw=2,
            color=color_of(name),
            label=f"{label_of(name)}: {(ratio > 1).mean():.0%} flagged",
        )
    ax.axvline(1, color=PALETTE["ink"], lw=1.2, ls=(0, (4, 3)))
    ax.annotate(
        f"threshold: {FALSE_ALARM_RATE:.1%} false alarms\nper component",
        (1, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-6, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        color=PALETTE["ink"],
    )
    ax.set_xscale("log")
    ax.set_xlim(grid_x[0], grid_x[-1])
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4], ["0.125", "0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel(
        "kNN distance divided by the threshold of the component, log scale (above 1 = Drift)"
    )
    ax.set_ylabel("Share of images above")
    ax.set_title("Share of images whose distance to the 5 nearest known images exceeds x")
    ax.legend(loc="upper right")

    for i, name in enumerate(PSEUDO_DRIFT):
        ax = fig.add_subplot(grid[1, i])
        for scores, auroc, role, method in (
            (maha, maha_auroc, "alternative", "Mahalanobis"),
            (knn, knn_auroc, "final", "kNN"),
        ):
            k, _, drift, _ = scores[name]
            fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(k)), np.ones(len(drift))], np.r_[k, drift])
            ax.plot(
                np.maximum(fpr, 1e-4),
                tpr,
                color=PALETTE[role],
                lw=2,
                label=f"{method} {auroc[name]:.3f}",
            )
        ax.axvline(FALSE_ALARM_RATE, color=PALETTE["muted"], lw=1, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title("Blur" if name == "blur" else name, fontsize=10)
        ax.set_xlabel("False alarms")
        if i == 0:
            ax.set_ylabel("Recall (ROC)")
        else:
            ax.tick_params(labelleft=False)
        ax.legend(loc="lower right", title="AUROC", title_fontsize=8.5, fontsize=8.5)

    fig.text(
        0.0,
        -0.03,
        "Legends give the AUROC of results/drift.json. The Mahalanobis curves are recomputed with "
        "a randomised PCA solver and stay within 0.005 of it; the kNN curves match exactly.",
        ha="left",
        color=PALETTE["ink_secondary"],
        fontsize=9,
    )
    at = reference["knn_crop"]
    blur_recall = at["blur"][f"at_{FALSE_ALARM_RATE}"]["recall"]
    mos_recall = at["Short circuit MOS"][f"at_{FALSE_ALARM_RATE}"]["recall"]
    wins = sum(knn_auroc[n] > maha_auroc[n] for n in PSEUDO_DRIFT)
    set_conclusion(
        fig,
        f"kNN has the best AUROC on {wins} of {len(PSEUDO_DRIFT)} pseudo-drift types; at 0.5% "
        f"false alarms it catches {blur_recall:.0%} of blur and {mos_recall:.0%} of "
        "Short circuit MOS",
    )
    save_fig(fig, "e_drift_detector")


# f. The GOOD decision ------------------------------------------------------------------------


def good_decision() -> None:
    rules = load("decision")["rules"]
    constants = load("leaderboard_constants")
    shares, pipeline = constants["shares"], constants["pipeline"]
    quantiles = (1.0, 0.75, 0.5, 0.25)
    names = ["None", "Closest 75%", "Closest 50%", "Closest 25%\n(submitted)"]
    gains, losses = [], []
    for q in quantiles:
        rule = rules[f"good_quantile_{q}"]
        gain = 100 * shares["good"] * rule["recall_GOOD"]
        gains.append(gain)
        losses.append(rule["expected_change_vs_never_good"] + gain)
    n = constants["evaluated_images"]
    measured = (pipeline["total_penalties"]["good"] - pipeline["total_penalties"]["no_good"]) / n
    expected = losses[-1] - gains[-1]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), width_ratios=(1.7, 1))
    x = np.arange(len(quantiles))
    ax.bar(x - 0.19, gains, width=0.36, color=PALETTE["final"], label="Expected gain")
    ax.bar(x + 0.19, losses, width=0.36, color=PALETTE["Drift"], label="Expected loss")
    for xi, value in zip(x - 0.19, gains, strict=True):
        ax.annotate(
            f"{value:.1f}", (xi, value), xytext=(0, 3), textcoords="offset points", ha="center"
        )
    for xi, value in zip(x + 0.19, losses, strict=True):
        ax.annotate(
            f"{value:.1f}", (xi, value), xytext=(0, 3), textcoords="offset points", ha="center"
        )
    ax.set_xticks(x, names)
    ax.set_xlabel("Proximity filter on known GOOD images of the component")
    ax.set_ylabel("Penalty points per image")
    ax.set_title("Before submitting: estimated on pseudo-drift and leaderboard shares")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")

    values = [-expected, -measured]
    bars = bx.bar([0, 1], values, width=0.55, color=[PALETTE["final"], PALETTE["Drift"]])
    for bar, value in zip(bars, values, strict=True):
        bx.annotate(
            f"{value:+.1f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4 if value > 0 else -14),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    bx.axhline(0, color=PALETTE["axis"], lw=1)
    bx.set_xticks([0, 1], ["Expected", "Measured on\nthe leaderboard"])
    bx.set_ylabel("Points saved per image by declaring GOOD")
    bx.set_title(
        f"After: score {pipeline['scores']['good']:.4f} instead of "
        f"{pipeline['scores']['no_good']:.4f}"
    )
    bx.grid(axis="x", visible=False)
    bx.set_ylim(min(values) * 1.25, max(values) * 4 + 1)
    set_conclusion(
        fig,
        f"Declaring GOOD was expected to save {-expected:.1f} points per image; one false GOOD "
        f"cost {measured:.1f}",
    )
    save_fig(fig, "f_good_decision")


# g. Argmax against minimum cost --------------------------------------------------------------


def argmax_vs_minimum_cost() -> None:
    baseline = load("baseline_logreg")
    best = baseline[baseline["best_by_log_loss"]]
    rows = [
        ("Logistic regression, argmax", best["argmax"]["mean_penalty"], PALETTE["alternative"]),
        (
            "All Missing (constant)",
            baseline["constant_missing"]["mean_penalty"],
            PALETTE["neutral"],
        ),
        (
            "Logistic regression, minimum cost",
            best["minimum_cost"]["mean_penalty"],
            PALETTE["final"],
        ),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for y, (_, value, color) in enumerate(rows):
        ax.barh(y, value, color=color, height=0.6)
        ax.annotate(
            f"{value:.1f}", (value, y), xytext=(6, 0), textcoords="offset points", va="center"
        )
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean penalty per image, 5-fold out of fold on the train set (lower is better)")
    ax.set_title(f"Same probabilities (log loss {best['log_loss']:.3f}), two ways to pick a label")
    ratio = rows[0][1] / rows[2][1]
    set_conclusion(fig, f"Picking the minimum-cost label divides the penalty by {ratio:.0f}")
    save_fig(fig, "g_argmax_vs_minimum_cost")


def main() -> None:
    set_style()
    score_ladder()
    metric_matrices()
    penalty_sources()
    good_decision()
    argmax_vs_minimum_cost()
    train, blurred, crop, blur_crop = load_crop_views()
    drift_detector(train, blurred, crop, blur_crop)
    embedding_map(train, blurred, crop, blur_crop)
    plt.close("all")


if __name__ == "__main__":
    main()
