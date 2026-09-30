"""
Build the windowed dataset, with every input restricted to its publication cutoff
(data/feature_classes.py; PAPER.md, Section 3.2).

One sample = one forecast origin = 12:00 CET on day D.
Target     = 24 hourly spreads for delivery day D+1.

Run:  python data/build_dataset.py --source ../Power-spread-forecasting/modelling_table.csv
Out:  data/dataset.npz, data/splits.json, data/manifest.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_classes as fc

TZ = "Europe/Berlin"          # the auction is defined in CET/CEST
HIST_HOURS = 168              # 7 days of context
HORIZON = 24                  # delivery day D+1
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SPLITS = {
    "train": [None, "2023-12-31"],
    "val":   ["2024-01-01", "2024-12-31"],
    "test":  ["2025-01-01", None],
}


def load_and_clean(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    groups = fc.audit(df.columns)                     # raises on unaudited columns
    df = df.drop(columns=fc.EXCLUDED_BROKEN)

    # Complete hourly grid, then interpolate the residual <0.4% gaps.
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    n_inserted = len(full) - len(df.index.intersection(full))
    df = df.reindex(full)
    df = df.interpolate(method="time", limit=3, limit_direction="both")

    df.index = df.index.tz_convert(TZ)
    return df, groups, n_inserted


def add_derived(df):
    """Features the literature flags as strong pre-auction signals (PLAN [L7])."""
    day = df.index.normalize()
    for z in ["DE", "FR", "PL"]:
        df[f"daymin_price_{z}"] = df.groupby(day)[f"price_{z}"].transform("min")
        df[f"daymax_price_{z}"] = df.groupby(day)[f"price_{z}"].transform("max")
        df[f"resid_load_fc_{z}"] = df[f"load_fc_{z}"] - df[f"wind_solar_fc_{z}"]
    missing = [c for c in fc.DERIVED_AUCTION + fc.DERIVED_FORECAST
               if c not in df.columns]
    if missing:
        raise ValueError(f"declared but not built: {missing}")
    return df


def build(df, groups):
    auc = [c for c in fc.AUCTION_PUBLISHED if c in df.columns]
    rea = [c for c in fc.REALISED if c in df.columns]
    ahd = [c for c in fc.FORECAST_AHEAD if c in df.columns]
    cal = [c for c in fc.CALENDAR if c in df.columns]

    by_day = {d: g for d, g in df.groupby(df.index.normalize())}
    days = sorted(by_day)

    X_auc, X_rea, X_ahd, X_cal, Y = [], [], [], [], []
    origins, prov, skipped = [], [], {"short_day": 0, "no_history": 0, "nan": 0}

    for i, D in enumerate(days[:-1]):
        D1 = days[i + 1]
        if (D1 - D).days != 1:
            continue
        tgt = by_day[D1]
        if len(tgt) != HORIZON:                       # DST 23h/25h days
            skipped["short_day"] += 1
            continue

        gate = D + pd.Timedelta(hours=12)             # 12:00 CET on day D

        # AUCTION_PUBLISHED: all of day D is known (published 12:45 on D-1).
        a_end = D + pd.Timedelta(hours=23)
        a_win = df.loc[:a_end, auc].iloc[-HIST_HOURS:]

        # REALISED: metered outturn, conservative 09:00 cut on day D.
        r_end = D + pd.Timedelta(hours=9)
        r_win = df.loc[:r_end, rea].iloc[-HIST_HOURS:]

        if len(a_win) < HIST_HOURS or len(r_win) < HIST_HOURS:
            skipped["no_history"] += 1
            continue

        ah = tgt[ahd].to_numpy(np.float32)
        ca = tgt[cal].to_numpy(np.float32)
        y = tgt[fc.TARGETS].to_numpy(np.float32)
        a = a_win.to_numpy(np.float32)
        r = r_win.to_numpy(np.float32)

        if any(np.isnan(v).any() for v in (a, r, ah, ca, y)):
            skipped["nan"] += 1
            continue

        X_auc.append(a); X_rea.append(r); X_ahd.append(ah); X_cal.append(ca); Y.append(y)
        origins.append(D)
        prov.append({
            "origin_day": str(D.date()),
            "gate_closure": gate.isoformat(),
            "auction_hist_start": a_win.index[0].isoformat(),
            "auction_hist_end": a_win.index[-1].isoformat(),
            "realised_hist_start": r_win.index[0].isoformat(),
            "realised_hist_end": r_win.index[-1].isoformat(),
            "target_start": tgt.index[0].isoformat(),
            "target_end": tgt.index[-1].isoformat(),
        })

    arrays = dict(
        X_auction=np.stack(X_auc), X_realised=np.stack(X_rea),
        X_ahead=np.stack(X_ahd), X_calendar=np.stack(X_cal), Y=np.stack(Y),
    )
    manifest = dict(
        tz=TZ, hist_hours=HIST_HOURS, horizon=HORIZON,
        cols=dict(auction=auc, realised=rea, ahead=ahd, calendar=cal,
                  target=fc.TARGETS),
        cutoffs=fc.CUTOFFS, skipped=skipped,
        n_samples=len(origins),
        origin_days=[str(d.date()) for d in origins],
        provenance=prov,
    )
    return arrays, manifest, origins


def assign_splits(origins):
    idx = {k: [] for k in SPLITS}
    for i, d in enumerate(origins):
        ds = str(d.date())
        for name, (lo, hi) in SPLITS.items():
            if (lo is None or ds >= lo) and (hi is None or ds <= hi):
                idx[name].append(i)
                break
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    a = ap.parse_args()

    df, groups, n_inserted = load_and_clean(a.source)
    df = add_derived(df)
    arrays, manifest, origins = build(df, groups)
    splits = assign_splits(origins)

    manifest["n_hours_inserted_on_regrid"] = int(n_inserted)
    manifest["splits"] = {k: [len(v), (str(origins[v[0]].date()) if v else None),
                              (str(origins[v[-1]].date()) if v else None)]
                          for k, v in splits.items()}

    np.savez_compressed(os.path.join(OUT_DIR, "dataset.npz"),
                        **arrays,
                        **{f"idx_{k}": np.array(v, dtype=np.int64) for k, v in splits.items()})
    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=2)
    json.dump({"spec": SPLITS, "counts": {k: len(v) for k, v in splits.items()}},
              open(os.path.join(OUT_DIR, "splits.json"), "w"), indent=2)

    print(f"samples: {manifest['n_samples']}   skipped: {manifest['skipped']}")
    for k, v in arrays.items():
        print(f"  {k:12s} {v.shape}")
    print("splits:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
