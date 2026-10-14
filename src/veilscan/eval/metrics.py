"""Detection metrics. Pure numpy so tests do not require sklearn."""

from __future__ import annotations

import numpy as np


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney
    # P(pos > neg) + 0.5 P(tie)
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    order = np.argsort(-y_score)
    yt = y_true[order]
    p = max(int(yt.sum()), 1)
    n = max(int((1 - yt).sum()), 1)
    fp = 0
    tp = 0
    best = 0.0
    for label in yt:
        if label == 1:
            tp += 1
        else:
            fp += 1
        if fp / n <= fpr_target:
            best = tp / p
        else:
            break
    return float(best)


def fpr_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    pred = np.asarray(y_score) >= float(threshold)
    n = int((y_true == 0).sum())
    if n <= 0:
        return float("nan")
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return float(fp / n)


def tpr_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    pred = np.asarray(y_score) >= float(threshold)
    p = int((y_true == 1).sum())
    if p <= 0:
        return float("nan")
    tp = int(((pred == 1) & (y_true == 1)).sum())
    return float(tp / p)


def cut_at_fpr(cover_scores: np.ndarray, fpr_target: float = 0.05) -> float:
    """Lowest threshold whose cover-FPR is <= fpr_target (quantile of covers)."""
    s = np.asarray(cover_scores, dtype=np.float64)
    if s.size == 0:
        return 0.5
    q = float(np.clip(1.0 - fpr_target, 0.0, 1.0))
    return float(np.quantile(s, q))


def f1_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    pred = (np.asarray(y_score) >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))
