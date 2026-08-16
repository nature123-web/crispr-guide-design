"""Metrics for guide efficiency prediction.

**Spearman correlation is the headline number**, not RMSE. Guide design is a
ranking problem: a researcher picks the top three or four guides for a target
and never uses the rest, so what matters is whether the ordering is right, not
whether the predicted efficiency is numerically accurate. Published CRISPR
models are compared on Spearman for exactly this reason.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy import stats

    if len(y_true) < 3 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).statistic)


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5,
                   top_fraction: float = 0.2) -> float:
    """Fraction of the top-``k`` predicted guides that are genuinely good.

    "Genuinely good" means in the top ``top_fraction`` by true efficiency. This
    is the metric that maps to the actual workflow: order four guides, how many
    work?
    """
    n = len(y_true)
    if n == 0 or k <= 0:
        return float("nan")
    k = min(k, n)
    n_good = max(1, int(n * top_fraction))
    good = set(np.argsort(-y_true)[:n_good].tolist())
    picked = np.argsort(-y_pred)[:k]
    return float(sum(1 for i in picked if i in good) / k)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Normalised discounted cumulative gain.

    Unlike precision@k this is graded rather than binary -- ranking a 0.9-
    efficiency guide first scores better than ranking a 0.7 one first, even
    though both are "good".
    """
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = min(k, n)

    def dcg(order):
        gains = y_true[order[:k]]
        discounts = np.log2(np.arange(2, k + 2))
        return float(np.sum(gains / discounts))

    ideal = dcg(np.argsort(-y_true))
    return dcg(np.argsort(-y_pred)) / ideal if ideal > 0 else float("nan")


def top_guide_efficiency(y_true: np.ndarray, y_pred: np.ndarray, k: int = 1
                         ) -> float:
    """Mean true efficiency of the guides the model would have you order."""
    if len(y_true) == 0:
        return float("nan")
    picked = np.argsort(-y_pred)[: min(k, len(y_true))]
    return float(np.mean(y_true[picked]))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "spearman": spearman(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "precision@5": precision_at_k(y_true, y_pred, 5),
        "ndcg@10": ndcg_at_k(y_true, y_pred, 10),
        "top1_efficiency": top_guide_efficiency(y_true, y_pred, 1),
        "top5_efficiency": top_guide_efficiency(y_true, y_pred, 5),
        "mean_efficiency": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def format_report(results: Dict[str, float], name: str = "model") -> str:
    lines = [f"{name}:"]
    for key in ("spearman", "pearson", "rmse", "mae", "precision@5", "ndcg@10",
                "top1_efficiency", "top5_efficiency", "mean_efficiency"):
        if key in results:
            lines.append(f"  {key:<18} {results[key]:.4f}")
    return "\n".join(lines)
