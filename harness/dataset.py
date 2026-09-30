"""Loader over the dataset built by data/build_dataset.py, plus the walk-forward
window logic.

Deliberately torch-free: the benchmarks need nothing but numpy, and keeping the
data layer independent means a bug here cannot be confused with a bug in the
deep learning models.
"""
import json
import os

import numpy as np
import pandas as pd

QUANTILES = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
MEDIAN_IDX = 3


class SpreadData:
    def __init__(self, data_dir):
        z = np.load(os.path.join(data_dir, "dataset.npz"))
        man = json.load(open(os.path.join(data_dir, "manifest.json")))
        self.X_auction = z["X_auction"]        # (N, 168, 22)  through end of day D
        self.X_realised = z["X_realised"]      # (N, 168, 11)  through 09:00 on D
        self.X_ahead = z["X_ahead"]            # (N, 24, 15)   forecasts for D+1
        self.X_calendar = z["X_calendar"]      # (N, 24, 7)    calendar for D+1
        self.Y = z["Y"]                        # (N, 24, 2)    spreads on D+1
        self.cols = man["cols"]
        self.origin = pd.to_datetime(man["origin_days"])
        self.n = len(self.Y)
        self.fixed = {k: z[f"idx_{k}"] for k in ("train", "val", "test")}

    # ---------------------------------------------------------------- features
    def flat(self, idx):
        """Everything, flattened. For the linear models."""
        parts = [self.X_auction[idx], self.X_realised[idx],
                 self.X_ahead[idx], self.X_calendar[idx]]
        return np.concatenate([p.reshape(len(idx), -1) for p in parts], axis=1)

    def lear_features(self, idx):
        """LEAR-style design matrix, hour-independent block (292 columns).

        Per origin: the target spreads over days D, D-1, D-2, D-6; the three zone
        prices over day D; that day's min/max per zone; the daily mean of the
        known-ahead covariates; and calendar.

        The covariates AT the target hour are added separately by
        `lear_features_hour`, following LEAR's own structure of one regression per
        target hour. Folding all 24 hours of all 15 covariates into one shared
        matrix instead would give 652 columns -- more than the ~620 training days a
        rolling 2-year window provides, which breaks the BIC noise-variance
        estimate. The walk-forward run found this; a fixed split with 1,794
        training days would not have.
        """
        A, H = self.X_auction[idx], self.X_ahead[idx]
        c = {n: i for i, n in enumerate(self.cols["auction"])}
        spreads = [c["spread_DE_FR"], c["spread_DE_PL"]]
        prices = [c["price_DE"], c["price_FR"], c["price_PL"]]
        mm = [c[f"day{k}_price_{z}"] for z in ("DE", "FR", "PL") for k in ("min", "max")]

        # history is ordered oldest -> newest; day D is the final 24 rows
        def day(offset):                      # offset 0 = day D, 1 = D-1, ...
            lo = A.shape[1] - 24 * (offset + 1)
            return A[:, lo:lo + 24, :]

        f = [day(k)[:, :, spreads].reshape(len(idx), -1) for k in (0, 1, 2, 6)]
        f.append(day(0)[:, :, prices].reshape(len(idx), -1))
        f.append(day(0)[:, -1, mm])
        f.append(H.mean(axis=1))
        f.append(self.X_calendar[idx][:, 0, :])          # calendar is per-day here
        return np.concatenate(f, axis=1)

    def lear_features_hour(self, idx, h):
        """The known-ahead covariates at target hour h (15 columns)."""
        return self.X_ahead[idx][:, h, :]

    def pooled(self, idx):
        """Compact representation: history summarised rather than flattened.

        mean/std/min/max/last per feature per stream, plus the full known-ahead
        block and one calendar row. 532 columns against 6,072 for `flat`. This is
        the numpy twin of nn_common.pool_sequence, so the sklearn MLP and the
        torch mlp_pool see the same information.
        """
        def pool(a):
            return np.concatenate([a.mean(1), a.std(1), a.min(1), a.max(1), a[:, -1, :]], axis=1)
        return np.concatenate([
            pool(self.X_auction[idx]), pool(self.X_realised[idx]),
            self.X_ahead[idx].reshape(len(idx), -1),
            self.X_calendar[idx][:, 0, :],
        ], axis=1)

    # ------------------------------------------------------------ walk-forward
    def retrain_points(self, start="2021-01-01", end="2026-07-01", freq="QS"):
        return pd.date_range(start, end, freq=freq)

    def windows(self, cut, policy="expanding", years=2, val_days=90):
        """Indices for one retrain point.

        train and val both lie strictly BEFORE `cut`; val is the last `val_days`
        of the training window, never data from after the cut. test is the next
        quarter. Returns (train_idx, val_idx, test_idx).
        """
        cut = pd.Timestamp(cut)
        before = self.origin < cut
        if policy == "rolling":
            before &= self.origin >= cut - pd.DateOffset(years=years)
        elif policy != "expanding":
            raise ValueError(policy)

        pool = np.where(before)[0]
        if len(pool) == 0:
            return pool, pool, np.array([], dtype=int)
        split = self.origin[pool] < (cut - pd.Timedelta(days=val_days))
        train, val = pool[split], pool[~split]

        test = np.where((self.origin >= cut) &
                        (self.origin < cut + pd.DateOffset(months=3)))[0]
        return train, val, test


def assert_no_future(data, cut, *idx_sets):
    """Guard against the one leak tests/test_no_leakage.py cannot see: a validation
    or training slice drawn from after the re-estimation date."""
    cut = pd.Timestamp(cut)
    for s in idx_sets:
        if len(s) and data.origin[s].max() >= cut:
            raise AssertionError(
                f"window contains origins at or after the retrain point {cut.date()}: "
                f"max origin {data.origin[s].max().date()}")
