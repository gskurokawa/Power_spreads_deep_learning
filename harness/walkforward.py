"""The walk-forward runner (PAPER.md, Section 5.1).

Every model is re-estimated at each of 23 quarterly dates from January 2021 to
July 2026, on all data before that date (the expanding window), and forecasts
each day of the following quarter. One row per (model, window, quarter, seed) is
appended to experiments/results.csv; the daily loss series and quantile forecasts
are saved to experiments/daily/.

Run:
    python harness/walkforward.py --models mlp_pool lstm transformer --seeds 0 1 2
    python harness/walkforward.py --models seasonal_naive lear

The benchmarks' principal results come from experiments/benchmarks.py, which adds
quantile regression post-processing to their point forecasts.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))

from dataset import SpreadData, assert_no_future
import metrics
import baselines


def registry(data):
    """name -> factory(seed) -> model with .fit/.predict.

    Baselines ignore the seed (they are deterministic); torch models take it.
    Built lazily so the numpy-only baselines still run where torch is absent.
    """
    reg = {name: (lambda cls=cls: (lambda seed=0: cls()))() for name, cls in baselines.ALL.items()}
    try:
        import torch  # noqa: F401
        import mlp
        from train import TorchModel
        for cfg_name, cfg in mlp.CONFIGS.items():
            reg[cfg_name] = (lambda c=cfg, n=cfg_name: (
                lambda seed=0: TorchModel(lambda: mlp.build(data, **c), n, seed=seed)))()
        for mod_name in ("transformer", "lstm"):
            try:
                mod = __import__(mod_name)
            except ImportError:
                continue
            for cfg_name, cfg in mod.CONFIGS.items():
                reg[cfg_name] = (lambda m=mod, c=cfg, n=cfg_name: (
                    lambda seed=0: TorchModel(lambda: m.build(data, **c), n, seed=seed)))()
    except ImportError:
        print("  [torch not available -- baselines only]")
    return reg

RESULTS = os.path.join(ROOT, "experiments", "results.csv")
DAILY = os.path.join(ROOT, "experiments", "daily")


def save_daily(name, policy, seed, days, pinball, mae, preds=None, truth=None):
    """Save the daily loss series and the quantile forecasts.

    results.csv holds one row per (model, window, quarter, seed), which is enough
    for the error tables but not for the Diebold-Mariano tests, which need one loss
    per forecast day (1,988 days rather than 23 quarterly means). The stored
    quantile forecasts also give the coverage table and the post-processing check.
    """
    os.makedirs(DAILY, exist_ok=True)
    extra = {}
    if preds is not None:
        # The quantile forecasts, (n_days, 24, 2, 7), and the realised values.
        extra["pred"] = np.asarray(preds, np.float32)
        extra["y"] = np.asarray(truth, np.float32)
    np.savez_compressed(
        os.path.join(DAILY, f"{name}__{policy}__seed{seed}.npz"),
        days=np.asarray(days, dtype="datetime64[D]"),
        pinball=np.asarray(pinball, float), mae=np.asarray(mae, float), **extra)


def append(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(RESULTS, mode="a", header=not os.path.exists(RESULTS), index=False)


def run_one(data, factory, name, train, val, test, tag, seed=0):
    np.random.seed(seed)
    t0 = time.time()
    model = factory(seed=seed).fit(data, train, val)
    fit_s = time.time() - t0

    pred = model.predict(data, test)
    run_one.last_pred = pred
    row = dict(tag, model=name, seed=seed,
               n_train=len(train), n_val=len(val), n_test=len(test),
               fit_seconds=round(fit_s, 2),
               n_params=getattr(model, "n_params", np.nan))
    row.update(metrics.summary(data.Y[test], pred))
    row["pinball_daily_mean"] = float(metrics.pinball_daily(data.Y[test], pred).mean())
    return row, metrics.pinball_daily(data.Y[test], pred), metrics.mae_daily(data.Y[test], pred)


def walk(data, model_names, policies=("expanding",), seeds=(0,), years=2):
    cuts = data.retrain_points()
    reg = registry(data)
    daily = {}
    for name in model_names:
        factory = reg[name]
        for policy in policies:
            per_seed = {sd: {"l": [], "m": [], "d": [], "p": [], "y": []} for sd in seeds}
            losses, maes, days = [], [], []
            for cut in cuts:
                train, val, test = data.windows(cut, policy=policy, years=years)
                if len(test) == 0 or len(train) < 200:
                    continue
                assert_no_future(data, cut, train, val)      # the walk-forward leak guard
                for seed in seeds:
                    row, dl, dm_ = run_one(
                        data, factory, name, train, val, test,
                        dict(protocol="walkforward", policy=policy,
                             retrain=str(pd.Timestamp(cut).date())), seed)
                    append(row)
                    per_seed[seed]["l"].append(dl)
                    per_seed[seed]["m"].append(dm_)
                    per_seed[seed]["d"].append(data.origin[test].values)
                    per_seed[seed]["p"].append(run_one.last_pred)
                    per_seed[seed]["y"].append(data.Y[test])
                    if seed == seeds[0]:
                        losses.append(dl); maes.append(dm_)
                        days.append(data.origin[test])
                print(f"  {name:15s} {policy:9s} {str(pd.Timestamp(cut).date())}  "
                      f"train={len(train):4d} test={len(test):3d}  "
                      f"MAE={row['spread_DE_FR_mae']:6.2f}  {row['fit_seconds']:.1f}s")
            for sd, acc in per_seed.items():
                if acc["l"]:
                    save_daily(name, policy, sd, np.concatenate(acc["d"]),
                               np.concatenate(acc["l"]), np.concatenate(acc["m"]),
                               np.concatenate(acc["p"]), np.concatenate(acc["y"]))
            if losses:
                daily[(name, policy)] = dict(
                    pinball=np.concatenate(losses), mae=np.concatenate(maes),
                    days=np.concatenate(days))
    return daily


def fixed(data, model_names, seeds=(0,)):
    reg = registry(data)
    out = {}
    for name in model_names:
        factory = reg[name]
        for seed in seeds:
            row, dl, dm_ = run_one(data, factory, name, data.fixed["train"], data.fixed["val"],
                                   data.fixed["test"], dict(protocol="fixed", policy="fixed",
                                                            retrain="2025-01-01"), seed)
            append(row)
            print(f"  {name:15s} fixed split  "
                  f"DE-FR MAE={row['spread_DE_FR_mae']:6.2f}  "
                  f"DE-PL MAE={row['spread_DE_PL_mae']:6.2f}  "
                  f"pinball={row['pinball']:.3f}  {row['fit_seconds']:.1f}s")
            if seed == seeds[0]:
                out[name] = dict(pinball=dl, mae=dm_)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--models", nargs="+",
                    default=["seasonal_naive", "lear", "mlp_pool", "lstm", "transformer"])
    ap.add_argument("--policies", nargs="+", default=["expanding"],
                    help="window policy: expanding (used in the paper) and/or rolling")
    ap.add_argument("--fixed-split", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()

    data = SpreadData(a.data_dir)
    print(f"{data.n} origins, {data.origin[0].date()} -> {data.origin[-1].date()}")
    if a.fixed_split:
        fixed(data, a.models, tuple(a.seeds))
    else:
        walk(data, a.models, policies=tuple(a.policies), seeds=tuple(a.seeds))


if __name__ == "__main__":
    main()
