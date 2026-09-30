"""Every table in PAPER.md, regenerated from the files on disk.

    python experiments/paper_tables.py

Inputs:
    data/                       dataset (Section 3.3)
    experiments/results.csv     quarterly results of the deep learning models, all
                                seeds (Sections 5.3, 6.1, 6.2, 6.4)
    experiments/daily/          daily losses and quantile forecasts of the deep
                                learning models, seed 0 (tests and coverage)
    experiments/daily_fixed/    benchmarks, written by benchmarks.py post
    experiments/daily_common/   identical post-processing, written by benchmarks.py common

Deep learning models: quarterly means are averaged over seeds 0, 1 and 2;
statistical tests and coverage use seed 0. The benchmarks are deterministic.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
from dataset import QUANTILES, SpreadData            # noqa: E402
from dmtest import dm_test                           # noqa: E402

EXP = os.path.join(ROOT, "experiments")
CRISIS = pd.date_range("2021-10-01", "2023-01-01", freq="QS")   # Oct 2021 - Mar 2023
DL = ["mlp_pool", "transformer", "lstm"]

# Paper name -> file holding its daily forecasts (seed 0 for deep learning models).
SOURCES = {
    "lear":        ("daily_fixed", "lear_q_qra"),
    "mlp_pool":    ("daily", "mlp_pool"),
    "naive":       ("daily_fixed", "seasonal_naive_q_qra"),
    "transformer": ("daily", "transformer"),
    "lstm":        ("daily", "lstm"),
}


def head(title):
    print("\n" + "=" * 86 + f"\n{title}\n" + "=" * 86)


def load(folder, stem):
    """Daily pinball loss and MAE per spread, plus the forecasts and realised values."""
    z = np.load(os.path.join(EXP, folder, f"{stem}__expanding__seed0.npz"))
    p, y = z["pred"].astype(float), z["y"].astype(float)
    e = y[..., None] - p
    q = QUANTILES.reshape(1, 1, 1, -1)
    df = pd.DataFrame({
        "pinball": np.maximum(q * e, (q - 1) * e).mean(axis=(1, 2, 3)),
        "mae_fr": np.abs(y[:, :, 0] - p[:, :, 0, 3]).mean(1),
        "mae_pl": np.abs(y[:, :, 1] - p[:, :, 1, 3]).mean(1),
    }, index=pd.to_datetime(z["days"]))
    return df, p, y


def quarterly(df):
    return df.groupby(df.index.to_period("Q").to_timestamp()).mean()


def dl_quarterly(name, seeds=(0, 1, 2)):
    """Quarterly means of a deep learning model, averaged over seeds."""
    r = pd.read_csv(os.path.join(EXP, "results.csv"))
    r = r[(r.protocol == "walkforward") & (r.policy == "expanding") &
          (r.model == name) & (r.seed.isin(seeds))]
    q = r.groupby("retrain")[["pinball", "spread_DE_FR_mae", "spread_DE_PL_mae"]].mean()
    q.columns = ["pinball", "mae_fr", "mae_pl"]
    q.index = pd.to_datetime(q.index)
    return q


def summary(q):
    crisis = q.index.isin(CRISIS)
    return {"Pinball (mean)": q.pinball.mean(), "Pinball (median quarter)": q.pinball.median(),
            "DE-FR MAE": q.mae_fr.mean(), "DE-PL MAE": q.mae_pl.mean(),
            "Crisis": q.pinball[crisis].mean(), "Other": q.pinball[~crisis].mean()}


def dm_row(a, b, la, lb):
    common = la.index.intersection(lb.index)
    r = dm_test(la.loc[common].values, lb.loc[common].values)
    lower = "neither at 5%" if r["p"] >= 0.05 else (a if r["stat"] < 0 else b)
    p = "< 0.0001" if r["p"] < 0.0001 else f"{r['p']:.4f}"
    print(f"  {a:12s} vs {b:12s}  {r['stat']:+7.2f}  p {p:>9s}  lower loss: {lower}"
          f"   (T = {len(common)})")


# ------------------------------------------------------------------ Section 3.3
def zero_share():
    head("Section 3.3  Proportion of hours in which the spread is exactly zero (%)")
    d = SpreadData(os.path.join(ROOT, "data"))
    sets = {"Full sample": np.arange(d.n),
            "Walk-forward test period": np.where(d.origin >= pd.Timestamp("2021-01-01"))[0],
            "Fixed-split test period": d.fixed["test"]}
    for label, idx in sets.items():
        fr, pl = (100 * (d.Y[idx, :, j] == 0).mean() for j in range(2))
        print(f"  {label:26s} DE-FR {fr:5.1f}   DE-PL {pl:5.1f}")


# ------------------------------------------------------------------ Section 5.3
def seed_threshold():
    """Median over models of the within-quarter standard deviation of MAE across
    seeds. As in the paper, this uses every model and window policy in results.csv
    and the seeds that cover every quarter for every model (seeds 0-2)."""
    head("Section 5.3  Seed-variability threshold (MAE)")
    r = pd.read_csv(os.path.join(EXP, "results.csv"))
    w = r[r.protocol == "walkforward"]
    per = w.groupby(["model", "seed"])["retrain"].nunique()
    full = w.retrain.nunique()
    complete = sorted(s for s in w.seed.unique()
                      if (per[per.index.get_level_values(1) == s] == full).all())

    def threshold(x, col):
        g = x.groupby(["model", "policy", "retrain"])[col]
        sd, n = g.std(), g.size()
        per_model = sd[n >= 2].groupby(level=0).median().dropna()
        return float(per_model[per_model > 0].median())

    ws = w[w.seed.isin(complete)]
    print(f"  seeds used: {complete}")
    print(f"  DE-FR {threshold(ws, 'spread_DE_FR_mae'):.2f}   "
          f"DE-PL {threshold(ws, 'spread_DE_PL_mae'):.2f}")
    print("  first k seeds, DE-FR:  " + "   ".join(
        f"k={k}: {threshold(w[w.seed < k], 'spread_DE_FR_mae'):.2f}" for k in (2, 3, 5, 8)))


# ------------------------------------------------------- Sections 6.1, 6.2, 6.4
def principal():
    daily = {n: load(*SOURCES[n])[0] for n in SOURCES}
    rows = {n: summary(dl_quarterly(n) if n in DL else quarterly(daily[n])) for n in SOURCES}
    tab = pd.DataFrame(rows).T

    head("Section 6.1  Principal results (quarterly re-estimation; benchmarks post-processed)")
    print(tab[["Pinball (mean)", "Pinball (median quarter)", "DE-FR MAE", "DE-PL MAE"]]
          .round(2).to_string())

    print("\n  Diebold-Mariano on daily pinball loss (seed 0); negative = first model lower")
    for a, b in [("lear", "mlp_pool"), ("lear", "transformer"), ("lear", "lstm"),
                 ("lear", "naive"), ("mlp_pool", "transformer"), ("mlp_pool", "lstm"),
                 ("mlp_pool", "naive"), ("transformer", "lstm"),
                 ("naive", "transformer"), ("naive", "lstm")]:
        dm_row(a, b, daily[a].pinball, daily[b].pinball)

    ql = quarterly(daily["lear"]).pinball
    for label, qm in [("seed 0", quarterly(daily["mlp_pool"]).pinball),
                      ("mean of seeds 0-2", dl_quarterly("mlp_pool").pinball)]:
        common = ql.index.intersection(qm.index)
        print(f"  LEAR has the lower quarterly pinball loss than the pooled MLP ({label}) in "
              f"{int((ql.loc[common] < qm.loc[common]).sum())} of {len(common)} quarters")
    print(f"  DE-FR MAE, transformer minus LEAR: "
          f"{tab.loc['transformer', 'DE-FR MAE'] - tab.loc['lear', 'DE-FR MAE']:.2f}")
    print(f"  DE-PL MAE, LEAR minus pooled MLP: "
          f"{tab.loc['lear', 'DE-PL MAE'] - tab.loc['mlp_pool', 'DE-PL MAE']:.2f}")

    head("Section 6.2  Deep learning models: pinball loss and parameter count")
    r = pd.read_csv(os.path.join(EXP, "results.csv"))
    for n in DL:
        params = int(r.loc[r.model == n, "n_params"].dropna().iloc[0])
        print(f"  {n:12s} {tab.loc[n, 'Pinball (mean)']:5.2f}   {params:,d}")
    t, m = tab.loc["transformer", "Pinball (mean)"], tab.loc["mlp_pool", "Pinball (mean)"]
    print(f"  transformer pinball loss above the pooled MLP: {100 * (t / m - 1):.1f}%")

    head("Section 6.4  Crisis quarters (Oct 2021 - Mar 2023) and other quarters: pinball")
    print(tab[["Crisis", "Other"]].round(2).to_string())


# ------------------------------------------------------------------ Section 6.3
def coverage():
    head("Section 6.3  Coverage: share of realised values at or below each quantile")
    rows = {"lear, post-processed": ("daily_fixed", "lear_q_qra"),
            "lear, fixed offsets": ("daily_fixed", "lear_q_orig"),
            "naive, post-processed": ("daily_fixed", "seasonal_naive_q_qra"),
            "mlp_pool": ("daily", "mlp_pool"),
            "transformer": ("daily", "transformer"),
            "lstm": ("daily", "lstm")}
    out = {}
    for label, src in rows.items():
        _, p, y = load(*src)
        out[label] = [(y <= p[..., k]).mean() for k in range(len(QUANTILES))]
    print(pd.DataFrame(out, index=[f"{q:.2f}" for q in QUANTILES]).T.round(3).to_string())


# ---------------------------------------------------------------- Section 6.5.1
def identical_post_processing():
    head("Section 6.5.1  Identical post-processing for all models "
         "(seed 0, 22 quarters from 2021 Q2)")
    start = pd.Timestamp("2021-04-01")
    lear = load("daily_common", "lear_q_qra")[0]
    print(f"  {'lear':12s} as trained      -   post-processed "
          f"{quarterly(lear).pinball.mean():5.2f}")
    learned = {}
    for n in DL:
        own = load("daily", n)[0]
        own = own[own.index >= start]
        learned[n] = own
        post = load("daily_common", f"{n}_qra")[0]
        print(f"  {n:12s} as trained {quarterly(own).pinball.mean():5.2f}   post-processed "
              f"{quarterly(post).pinball.mean():5.2f}")
    print()
    dm_row("lear", "mlp_pool", lear.pinball, learned["mlp_pool"].pinball)


# ---------------------------------------------------------------- Section 6.5.2
def fixed_offsets():
    head("Section 6.5.2  Benchmarks: fixed residual offsets against quantile regression")
    for label, stem in [("lear", "lear_q"), ("naive", "seasonal_naive_q")]:
        fixed = quarterly(load("daily_fixed", f"{stem}_orig")[0]).pinball.mean()
        qra = quarterly(load("daily_fixed", f"{stem}_qra")[0]).pinball.mean()
        print(f"  {label:12s} fixed offsets {fixed:5.2f}   quantile regression {qra:5.2f}")


# ---------------------------------------------------------------- Section 6.5.3
def daily_lear():
    head("Section 6.5.3  LEAR re-estimated quarterly and daily (post-processed)")
    q = load("daily_fixed", "lear_q_qra")[0]
    d = load("daily_fixed", "lear_d_qra")[0]
    print(f"  quarterly {quarterly(q).pinball.mean():5.2f}   daily {quarterly(d).pinball.mean():5.2f}")
    dm_row("lear daily", "lear qtrly", d.pinball, q.pinball)


def main():
    zero_share()
    seed_threshold()
    principal()
    coverage()
    identical_post_processing()
    fixed_offsets()
    daily_lear()


if __name__ == "__main__":
    main()
