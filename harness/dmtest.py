"""Diebold-Mariano test of equal predictive accuracy (PAPER.md, Section 5.2).

Input is always a daily loss series, one value per forecast origin: the 24 hourly
losses of one day come from a single clearing event and are not independent.
The long-run variance of the loss differential is estimated by the Newey-West
estimator, with the Harvey-Leybourne-Newbold small-sample correction.
"""
import numpy as np
from scipy import stats


def newey_west_lrv(d, lag=None):
    """Long-run variance, Bartlett kernel. lag=None uses 4*(T/100)^(2/9)."""
    d = np.asarray(d, float)
    T = len(d)
    if lag is None:
        lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    g0 = (x @ x) / T
    s = g0
    for k in range(1, lag + 1):
        gk = (x[k:] @ x[:-k]) / T
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return max(s, 1e-12), lag


def dm_test(loss_a, loss_b, h=1, hln=True, lag=None):
    """Diebold-Mariano on daily losses, Newey-West variance.

    Negative statistic => model A has lower loss => A is better.
    `hln` applies the Harvey-Leybourne-Newbold small-sample correction and uses
    the t distribution rather than the normal.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T = len(d)
    lrv, used_lag = newey_west_lrv(d, lag)
    stat = d.mean() / np.sqrt(lrv / T)

    if hln:
        k = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
        stat *= k
        p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    else:
        p = 2 * (1 - stats.norm.cdf(abs(stat)))

    return {"stat": float(stat), "p": float(p), "mean_diff": float(d.mean()),
            "T": int(T), "nw_lag": int(used_lag),
            "better": "A" if d.mean() < 0 else "B"}
