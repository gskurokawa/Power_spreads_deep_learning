"""Benchmark forecasts: point forecasts for LEAR and the naive forecast, daily
re-estimation of LEAR, and quantile regression post-processing.

LEAR and the naive forecast produce one point forecast per hour. They are given a
predictive distribution here by quantile regression post-processing (PAPER.md,
Section 4.4): for each spread and quantile level, the realised spread is regressed
on the model's own point forecast f, on |f|, and on 24 hour indicators, using the
model's out-of-sample forecasts for the 365 days before each quarterly
re-estimation date. The seven predicted quantiles are sorted into ascending order.

Everything is walk-forward. At each quarterly cut the post-processing is fitted only
on forecasts, and realised values, for days strictly before the cut. Point forecasts
from 2020 Q4 give the first test quarter (2021 Q1) one quarter of calibration data,
rising to a full year by 2021 Q4. These warm-up days are never scored.

Stages:
    python experiments/benchmarks.py points      # quarterly LEAR and naive (minutes)
    python experiments/benchmarks.py daily --worker 0 --nworkers 2   # daily LEAR
    python experiments/benchmarks.py daily --worker 1 --nworkers 2   # (hours; restartable)
    python experiments/benchmarks.py merge       # combine the daily workers
    python experiments/benchmarks.py post        # post-process and score (Sections 6.1-6.5)
    python experiments/benchmarks.py common      # same post-processing for every model (6.5.1)

Output files in experiments/daily_fixed/:
    <model>_q_qra    quarterly re-estimation, quantile regression post-processing
    <model>_q_orig   quarterly re-estimation, fixed residual offsets (Section 6.5.2)
    lear_d_qra       daily re-estimation of LEAR (Section 6.5.3)
"""
import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "models"))
from dataset import QUANTILES, SpreadData          # noqa: E402
import baselines                                   # noqa: E402
import metrics                                     # noqa: E402

warnings.filterwarnings("ignore")                  # LARS early-stopping notices

POINTS = os.path.join(ROOT, "experiments", "points")
OUT = os.path.join(ROOT, "experiments", "daily_fixed")
WARMUP_START = "2020-01-01"   # quarterly baselines: cheap, so a full warm-up year
DAILY_START = "2020-10-01"    # daily LEAR: fits with under ~600 training days are
                              # slow (Lasso path near n = p), so warm-up is 1 quarter
CAL_START = "2020-10-01"      # common start for post-processor calibration data, so
                              # quarterly and daily variants see identical windows
EVAL_START = "2021-01-01"
CAL_DAYS = 365           # calibration window for the post-processor
DAILY_WINDOW = 4 * 364   # longest of the four epftoolbox calibration windows


# ---------------------------------------------------------------- point stage
def quarterly_points(data, name):
    """Re-estimate at each quarter start from 2020, as harness/walkforward.py does.
    Keeps the point forecast, and the fixed-residual-offset quantiles produced by
    the model itself (models/baselines.py), which Section 6.5.2 reports."""
    cuts = pd.date_range(WARMUP_START, "2026-07-01", freq="QS")
    days, pts, orig, ys = [], [], [], []
    for cut in cuts:
        tr, va, te = data.windows(cut, policy="expanding")
        if len(te) == 0:
            continue
        m = baselines.ALL[name]().fit(data, tr, va)
        pts.append(m._point(data, te))
        orig.append(m.predict(data, te))
        days.append(data.origin[te].values)
        ys.append(data.Y[te])
        print(f"  {name:15s} {cut.date()}  train={len(tr):4d} test={len(te):3d}")
    np.savez_compressed(os.path.join(POINTS, f"{name}__quarterly.npz"),
                        days=np.concatenate(days).astype("datetime64[D]"),
                        point=np.concatenate(pts), orig=np.concatenate(orig),
                        y=np.concatenate(ys))


def daily_points(data, worker, nworkers):
    """Re-estimate LEAR every day on the preceding four years, or on all history
    if shorter (Section 6.5.3). Days are interleaved across workers, and each
    worker saves a checkpoint every 20 fits so that it can be restarted."""
    target = np.where(data.origin >= pd.Timestamp(DAILY_START))[0]
    mine = target[worker::nworkers]
    ck = os.path.join(POINTS, f"lear__daily_w{worker}.npz")
    done, pts = {}, {}
    if os.path.exists(ck):
        z = np.load(ck)
        for i, p in zip(z["idx"], z["point"]):
            pts[int(i)] = p
        print(f"  resuming worker {worker}: {len(pts)} of {len(mine)} done")

    def save():
        k = sorted(pts)
        np.savez_compressed(ck, idx=np.array(k), point=np.stack([pts[i] for i in k]))

    t0, n_new = time.time(), 0
    for i in mine:
        if int(i) in pts:
            continue
        t = data.origin[i]
        tr = np.where((data.origin < t) &
                      (data.origin >= t - pd.Timedelta(days=DAILY_WINDOW)))[0]
        m = baselines.LEAR().fit(data, tr, np.array([], dtype=int))
        pts[int(i)] = m._point(data, np.array([i]))[0]
        n_new += 1
        if n_new % 20 == 0:
            save()
            rate = (time.time() - t0) / n_new
            left = sum(1 for j in mine if int(j) not in pts)
            print(f"  worker {worker}: {len(pts)}/{len(mine)}  {rate:.1f}s/fit  "
                  f"~{left * rate / 3600:.1f}h left  (at {t.date()})", flush=True)
    save()
    print(f"  worker {worker} complete: {len(pts)} fits")


def merge_daily(data):
    idx, pts = [], []
    for f in sorted(glob.glob(os.path.join(POINTS, "lear__daily_w*.npz"))):
        z = np.load(f)
        idx.append(z["idx"]); pts.append(z["point"])
    idx = np.concatenate(idx); pts = np.concatenate(pts)
    o = np.argsort(idx); idx, pts = idx[o], pts[o]
    expected = np.where(data.origin >= pd.Timestamp(DAILY_START))[0]
    missing = np.setdiff1d(expected, idx)
    if len(missing):
        raise SystemExit(f"{len(missing)} days missing; workers not finished")
    np.savez_compressed(os.path.join(POINTS, "lear__daily.npz"),
                        days=data.origin[idx].values.astype("datetime64[D]"),
                        point=pts, y=data.Y[idx])
    print(f"  merged {len(idx)} daily LEAR forecasts")


# ---------------------------------------------------------- post-processing
def _design(f):
    """Rows = (day, hour) for one spread. Columns: 24 hour dummies, f, |f|."""
    n = f.shape[0]
    H = np.tile(np.eye(24), (n, 1))
    ff = f.reshape(-1, 1)
    return np.hstack([H, ff, np.abs(ff)])


def post_process(days, point, y, cal_start=CAL_START, eval_start=EVAL_START):
    """Quantile regression post-processing (PAPER.md, Section 4.4).

    Returns quantile forecasts for every day from eval_start. At each quarterly cut,
    for each spread and quantile level, the realised spread is regressed on 24 hour
    indicators, f and |f| over the preceding CAL_DAYS days, and the fitted
    regression is applied to the point forecasts of the following quarter."""
    from sklearn.linear_model import QuantileRegressor
    days = pd.to_datetime(days)
    out = np.full(point.shape + (len(QUANTILES),), np.nan)
    cuts = pd.date_range(eval_start, "2026-07-01", freq="QS")
    for cut in cuts:
        te = np.where((days >= cut) & (days < cut + pd.DateOffset(months=3)))[0]
        if len(te) == 0:
            continue
        cal = np.where((days < cut) & (days >= pd.Timestamp(cal_start)) &
                       (days >= cut - pd.Timedelta(days=CAL_DAYS)))[0]
        for j in range(2):
            Xc, yc = _design(point[cal, :, j]), y[cal, :, j].ravel()
            Xt = _design(point[te, :, j])
            for k, q in enumerate(QUANTILES):
                qr = QuantileRegressor(quantile=q, alpha=0.0, fit_intercept=False,
                                       solver="highs").fit(Xc, yc)
                out[te, :, j, k] = qr.predict(Xt).reshape(len(te), 24)
    return np.sort(out, axis=-1)          # rearrangement: no quantile crossing


def score_and_save(name, days, pred, y, eval_start=EVAL_START, out=None):
    days = pd.to_datetime(days)
    keep = days >= pd.Timestamp(eval_start)
    d, p, t = days[keep], pred[keep], y[keep]
    assert not np.isnan(p).any(), name
    np.savez_compressed(os.path.join(out or OUT, f"{name}__expanding__seed0.npz"),
                        days=d.values.astype("datetime64[D]"),
                        pinball=metrics.pinball_daily(t, p), mae=metrics.mae_daily(t, p),
                        mae_fr=np.abs(t[:, :, 0] - p[:, :, 0, 3]).mean(1),
                        mae_pl=np.abs(t[:, :, 1] - p[:, :, 1, 3]).mean(1),
                        pred=p.astype(np.float32), y=t.astype(np.float32))
    print(f"  {name:28s} days={len(d)}  pinball={metrics.pinball(t, p):.3f}")


def run_post(only=None):
    os.makedirs(OUT, exist_ok=True)
    sources = {}
    for f in sorted(glob.glob(os.path.join(POINTS, "*__quarterly.npz"))):
        sources[os.path.basename(f).split("__")[0] + "_q"] = np.load(f)
    f = os.path.join(POINTS, "lear__daily.npz")
    if os.path.exists(f):
        sources["lear_d"] = np.load(f)
    for base, z in sources.items():
        if only and base not in only:
            continue
        if "orig" in z.files:                   # fixed residual offsets (Section 6.5.2)
            score_and_save(f"{base}_orig", z["days"], z["orig"], z["y"])
        t0 = time.time()
        p = post_process(z["days"], z["point"], z["y"])
        score_and_save(f"{base}_qra", z["days"], p, z["y"])
        print(f"      ({time.time() - t0:.0f}s)")


def run_common(only=None):
    """Identical post-processing for every model (PAPER.md, Section 6.5.1).

    LEAR (quarterly and daily) and the three deep learning models (seed 0) have
    their central forecast -- the point forecast for LEAR, the predicted median for
    the deep learning models -- post-processed identically. The deep learning
    forecasts begin in 2021, so calibration starts 2021-01-01 for every model and
    scoring starts 2021-04-01 (22 quarters).
    """
    cal0, ev0 = "2021-01-01", "2021-04-01"
    out = os.path.join(ROOT, "experiments", "daily_common")
    os.makedirs(out, exist_ok=True)
    src = {}
    for m in ("mlp_pool", "transformer", "lstm"):
        z = np.load(os.path.join(ROOT, "experiments", "daily", f"{m}__expanding__seed0.npz"))
        src[m] = (z["days"], z["pred"][..., 3].astype(float), z["y"].astype(float))
    for m, f in (("lear_q", "lear__quarterly.npz"), ("lear_d", "lear__daily.npz")):
        if os.path.exists(os.path.join(POINTS, f)):
            z = np.load(os.path.join(POINTS, f))
            src[m] = (z["days"], z["point"], z["y"])
    for m, (d, f, y) in src.items():
        if only and m not in only:
            continue
        t0 = time.time()
        p = post_process(d, f, y, cal_start=cal0, eval_start=ev0)
        score_and_save(f"{m}_qra", d, p, y, eval_start=ev0, out=out)
        print(f"      ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["points", "daily", "merge", "post", "common"])
    ap.add_argument("--models", nargs="+", default=["seasonal_naive", "lear"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--sources", nargs="+", default=None,
                    help="post/common stages: restrict to these sources, e.g. lear_q lear_d")
    a = ap.parse_args()
    os.makedirs(POINTS, exist_ok=True)
    data = SpreadData(os.path.join(ROOT, "data"))
    if a.stage == "points":
        for m in a.models:
            quarterly_points(data, m)
    elif a.stage == "daily":
        daily_points(data, a.worker, a.nworkers)
    elif a.stage == "merge":
        merge_daily(data)
    elif a.stage == "common":
        run_common(a.sources)
    else:
        run_post(a.sources)


if __name__ == "__main__":
    main()
