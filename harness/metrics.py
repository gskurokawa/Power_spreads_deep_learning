"""Scoring. Every model is judged by exactly these functions.

Shapes throughout: y_true (n, 24, 2), y_pred (n, 24, 2, Q).
"""
import numpy as np

from dataset import MEDIAN_IDX, QUANTILES


def pinball(y_true, y_pred, quantiles=QUANTILES):
    """Mean pinball loss. The training objective and the primary metric."""
    e = y_true[..., None] - y_pred
    q = quantiles.reshape(1, 1, 1, -1)
    return float(np.maximum(q * e, (q - 1) * e).mean())


def pinball_daily(y_true, y_pred, quantiles=QUANTILES):
    """One loss value per origin day -- the unit of observation for DM tests.

    Testing on 24*n hourly errors when there are only n independent forecast
    origins overstates the sample by 24x. No HAC correction repairs that.
    """
    e = y_true[..., None] - y_pred
    q = quantiles.reshape(1, 1, 1, -1)
    return np.maximum(q * e, (q - 1) * e).mean(axis=(1, 2, 3))


def point(y_pred):
    """Conditional median, for comparability with point-forecast benchmarks."""
    return y_pred[..., MEDIAN_IDX]


def mae_daily(y_true, y_pred):
    return np.abs(y_true - point(y_pred)).mean(axis=(1, 2))


def summary(y_true, y_pred, target_names=("spread_DE_FR", "spread_DE_PL")):
    """Headline numbers, overall and split by congestion state.

    The congestion split matters because a model can score well on a target that
    is ~30% exact zeros simply by predicting near zero everywhere.
    """
    p = point(y_pred)
    out = {"pinball": pinball(y_true, y_pred)}
    for j, name in enumerate(target_names):
        yt, yp = y_true[:, :, j], p[:, :, j]
        err = yt - yp
        zero = yt == 0
        out[f"{name}_mae"] = float(np.abs(err).mean())
        out[f"{name}_rmse"] = float(np.sqrt((err ** 2).mean()))
        out[f"{name}_mae_uncongested"] = float(np.abs(err[zero]).mean()) if zero.any() else np.nan
        out[f"{name}_mae_congested"] = float(np.abs(err[~zero]).mean()) if (~zero).any() else np.nan
        out[f"{name}_zero_share"] = float(zero.mean())
    return out


def coverage(y_true, y_pred, quantiles=QUANTILES):
    """Empirical coverage per quantile. Should track the nominal level; if it
    does not, good pinball loss is hiding a miscalibrated predictive interval."""
    return {float(q): float((y_true <= y_pred[..., i]).mean())
            for i, q in enumerate(quantiles)}
