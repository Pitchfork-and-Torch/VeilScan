import numpy as np

from veilscan.eval.metrics import auc_roc, f1_at, tpr_at_fpr


def test_auc_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_roc(y, s) == 1.0


def test_auc_random_ish() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.4, 0.41, 0.42, 0.43])
    a = auc_roc(y, s)
    assert 0.0 <= a <= 1.0


def test_tpr_at_fpr() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert tpr_at_fpr(y, s, 0.05) >= 0.0
    assert f1_at(y, s, 0.5) > 0.9
