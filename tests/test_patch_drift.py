import numpy as np

from valeo_qc.patch_drift import out_of_fold_patch_scores, patch_scores


def test_out_of_fold_patch_scores_do_not_match_an_image_with_itself():
    # distinct random patches: only a leak of the image itself would give a zero distance
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(40, 16, 32))
    libs = np.array(["Die01"] * 40)
    folds = np.tile(np.arange(5), 8)
    assert out_of_fold_patch_scores(patches, libs, folds).min() > 0.1


def test_one_unknown_patch_raises_the_score_of_its_image():
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(50, 8))
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    known = np.repeat(bank[:16][None], 2, axis=0)
    known[1, 0] = -bank[0] - bank[1]
    scores = patch_scores(bank, known, top=1)
    assert scores[0] < 1e-5 < scores[1]
