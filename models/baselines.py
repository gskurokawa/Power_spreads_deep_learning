"""Benchmark models: the naive forecast and LEAR (PAPER.md, Sections 4.2 and 4.1).

Every model exposes the same two methods, so the walk-forward runner treats the
benchmarks and the deep learning models identically:

    fit(data, train_idx, val_idx)
    predict(data, idx) -> (n, 24, 2, Q)

`predict` returns quantiles formed by fixed residual offsets: the empirical
quantiles of the errors on the validation slice, added to the point forecast.
These are the "fixed residual offsets" of Section 6.5.2. The principal results
instead give the point forecasts (`_point`) a predictive distribution by quantile
regression post-processing, in experiments/benchmarks.py.
"""
import numpy as np
from sklearn.linear_model import LassoCV, LassoLarsIC
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from dataset import QUANTILES


class ResidualQuantileMixin:
    """Fixed residual offsets: point forecast plus the empirical quantiles of the
    validation-slice errors, estimated separately for each hour and spread.

    With a 90-day validation slice there are about 90 errors per hour and spread,
    so the 0.05 and 0.95 offsets rest on a handful of observations each.
    """

    def _fit_residual_quantiles(self, y_true, y_point):
        r = y_true - y_point                                  # (n, 24, 2)
        if len(r) < 20:                                       # too few to bother
            self.res_q = np.zeros((24, 2, len(QUANTILES)))
            self.res_q[:] = np.quantile(r.reshape(-1), QUANTILES) if len(r) else 0.0
            return
        self.res_q = np.quantile(r, QUANTILES, axis=0).transpose(1, 2, 0)

    def _apply(self, y_point):
        return y_point[..., None] + self.res_q[None, ...]


class SeasonalNaive(ResidualQuantileMixin):
    """The naive forecast (PAPER.md, Section 4.2); code identifier seasonal_naive.

    Tuesday to Friday copy the same hour of day D; Saturday, Sunday and Monday copy
    the same hour one week before the delivery day. Both are available at the
    forecast origin: day D's spreads were published at 12:45 on day D-1, and the
    earlier day lies within the history window."""

    name = "seasonal_naive"

    def _point(self, data, idx):
        c = {n: i for i, n in enumerate(data.cols["auction"])}
        sp = [c["spread_DE_FR"], c["spread_DE_PL"]]
        A = data.X_auction[idx]
        prev_day = A[:, -24:, :][:, :, sp]                    # day D
        prev_week = A[:, 0:24, :][:, :, sp]                   # day D-6
        dow = data.X_calendar[idx][:, 0, data.cols["calendar"].index("dayofweek")]
        use_week = np.isin(dow, [0, 5, 6])[:, None, None]     # Mon, Sat, Sun
        return np.where(use_week, prev_week, prev_day)

    def fit(self, data, train_idx, val_idx):
        if len(val_idx):
            self._fit_residual_quantiles(data.Y[val_idx], self._point(data, val_idx))
        else:
            self._fit_residual_quantiles(data.Y[train_idx], self._point(data, train_idx))
        return self

    def predict(self, data, idx):
        return self._apply(self._point(data, idx))


class LEAR(ResidualQuantileMixin):
    """Lasso Estimated AutoRegressive -- the standard linear benchmark in
    electricity price forecasting (Lago et al. 2021).

    One Lasso per (target hour, spread), 48 in total, with the BIC-selected
    penalty from LassoLarsIC. The target is asinh-transformed with a median/MAD
    scale, the variance-stabilising step LEAR uses to stop price spikes dominating
    the fit -- asinh rather than log because spreads go negative.
    """

    name = "lear"

    def _design(self, data, idx, h):
        return np.concatenate([data.lear_features(idx),
                               data.lear_features_hour(idx, h)], axis=1)

    def _estimator(self, n, p):
        """LassoLarsIC's BIC needs a noise-variance estimate, which requires
        n > p. Fall back to cross-validated Lasso when a short rolling window
        makes that false, rather than silently crashing mid-run."""
        if n > p + 1:
            return LassoLarsIC(criterion="bic")
        return LassoCV(cv=5, n_alphas=30, max_iter=5000, random_state=0)

    def fit(self, data, train_idx, val_idx):
        Y = data.Y[train_idx]
        self.med = np.median(Y, axis=0)                                   # (24, 2)

        # Scale for the asinh transform. A plain MAD scale COLLAPSES here: with
        # 40% of hours at exactly zero the median absolute deviation falls to
        # ~0.01 for some cells, which pushes z out to +-9, and since the inverse
        # is sinh, a 2-unit error in z becomes an error of 1e5 EUR/MWh. The
        # walk-forward run hit exactly this in the 2021-22 quarters (MAE ~1e13).
        # Flooring the scale at half the standard deviation keeps the transform
        # well conditioned; asinh still does the tail-taming it is there for.
        mad = 1.4826 * np.median(np.abs(Y - self.med), axis=0)
        self.scale = np.maximum(mad, 0.5 * Y.std(axis=0))
        self.scale = np.where(self.scale > 1e-3, self.scale, 1.0)

        z_all = np.arcsinh((Y - self.med) / self.scale)
        self.z_lo = z_all.min(axis=0) - 0.5      # belt and braces: never invert
        self.z_hi = z_all.max(axis=0) + 0.5      # sinh outside the observed range

        self.scalers, self.models = [], []
        for h in range(24):
            X = self._design(data, train_idx, h)
            sc = StandardScaler().fit(X)
            Xs = sc.transform(X)
            self.scalers.append(sc)
            row = []
            for j in range(2):
                z = np.arcsinh((Y[:, h, j] - self.med[h, j]) / self.scale[h, j])
                row.append(self._estimator(*Xs.shape).fit(Xs, z))
            self.models.append(row)

        ref = val_idx if len(val_idx) else train_idx
        self._fit_residual_quantiles(data.Y[ref], self._point(data, ref))
        return self

    def _point(self, data, idx):
        out = np.zeros((len(idx), 24, 2))
        for h in range(24):
            Xs = self.scalers[h].transform(self._design(data, idx, h))
            for j in range(2):
                z = np.clip(self.models[h][j].predict(Xs), self.z_lo[h, j], self.z_hi[h, j])
                out[:, h, j] = np.sinh(z) * self.scale[h, j] + self.med[h, j]
        return out

    def predict(self, data, idx):
        return self._apply(self._point(data, idx))


ALL = {m.name: m for m in (SeasonalNaive, LEAR)}
