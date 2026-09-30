"""
Phase 0 done-when gate: fail loudly if anything the model can see was published
at or after gate closure for its own origin day.

Deliberately INDEPENDENT of build_dataset.py -- it reloads the source CSV and
re-derives every value from the timestamps recorded in the manifest, rather than
trusting the builder's internal slicing. A bug in the builder therefore shows up
here as a mismatch, not as an agreed-upon error.

Run:  python tests/test_no_leakage.py --source ../Power-spread-forecasting/modelling_table.csv
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
import feature_classes as fc

TZ = "Europe/Berlin"
FAILS, CHECKS = [], 0


def check(name, cond, detail=""):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(f"{name}: {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--n-spot-checks", type=int, default=40)
    a = ap.parse_args()

    man = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    z = np.load(os.path.join(ROOT, "data", "dataset.npz"))
    prov = man["provenance"]
    cols = man["cols"]

    # ---------------------------------------------------------------- 1. classes
    for c in cols["auction"]:
        check("history block purity", fc.CLASS_OF.get(c) == "AUCTION_PUBLISHED",
              f"{c} is {fc.CLASS_OF.get(c)}, not AUCTION_PUBLISHED")
    for c in cols["realised"]:
        check("history block purity", fc.CLASS_OF.get(c) == "REALISED",
              f"{c} is {fc.CLASS_OF.get(c)}, not REALISED")
    for c in cols["ahead"]:
        check("ahead block purity", fc.CLASS_OF.get(c) == "FORECAST_AHEAD",
              f"{c} in ahead block but classed {fc.CLASS_OF.get(c)}")
    for c in cols["calendar"]:
        check("calendar block purity", fc.CLASS_OF.get(c) == "CALENDAR",
              f"{c} in calendar block but classed {fc.CLASS_OF.get(c)}")

    # The cardinal rule: nothing metered or auction-cleared may describe D+1.
    forbidden = set(cols["auction"]) | set(cols["realised"])
    check("no outturn in ahead block",
          not (forbidden & set(cols["ahead"] + cols["calendar"])),
          str(forbidden & set(cols["ahead"] + cols["calendar"])))

    # -------------------------------------------------------------- 2. timestamps
    for p in prov:
        D = pd.Timestamp(p["origin_day"], tz=TZ)
        gate = pd.Timestamp(p["gate_closure"])
        a_end = pd.Timestamp(p["auction_hist_end"])
        r_end = pd.Timestamp(p["realised_hist_end"])
        t0 = pd.Timestamp(p["target_start"])
        t1 = pd.Timestamp(p["target_end"])

        check("gate is 12:00 on D", gate == D + pd.Timedelta(hours=12), p["origin_day"])

        # Auction outputs for day D were published 12:45 CET on D-1, so the whole
        # of day D is legitimately known -- but not one hour of D+1.
        check("auction history stops at end of D",
              a_end <= D + pd.Timedelta(hours=23), f"{p['origin_day']} {a_end}")
        check("auction history never touches target day",
              a_end < t0, f"{p['origin_day']} {a_end} >= {t0}")

        # Metered data is cut before gate closure with a safety margin.
        check("realised history ends before gate closure",
              r_end < gate, f"{p['origin_day']} {r_end} >= {gate}")
        check("realised history respects the 09:00 cut",
              r_end <= D + pd.Timedelta(hours=9), f"{p['origin_day']} {r_end}")

        # Target is exactly delivery day D+1.
        # NB: date arithmetic, not timedelta. On the October long day the local
        # day is 25h, so D + 24h lands at 23:00 rather than midnight.
        check("target starts at 00:00 the day after D",
              t0.hour == 0 and (t0.date() - D.date()).days == 1,
              f"{p['origin_day']} {t0}")
        check("target spans 24h", (t1 - t0) == pd.Timedelta(hours=23),
              f"{p['origin_day']} {t0}..{t1}")

    # ------------------------------------------------- 3. values match the source
    df = pd.read_csv(a.source, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full).interpolate(method="time", limit=3, limit_direction="both")
    df.index = df.index.tz_convert(TZ)
    day = df.index.normalize()
    for zn in ["DE", "FR", "PL"]:
        df[f"daymin_price_{zn}"] = df.groupby(day)[f"price_{zn}"].transform("min")
        df[f"daymax_price_{zn}"] = df.groupby(day)[f"price_{zn}"].transform("max")
        df[f"resid_load_fc_{zn}"] = df[f"load_fc_{zn}"] - df[f"wind_solar_fc_{zn}"]

    rng = np.random.default_rng(0)
    picks = rng.choice(len(prov), size=min(a.n_spot_checks, len(prov)), replace=False)
    for i in picks:
        p = prov[i]
        for block, key, start, end in [
            ("auction", "X_auction", p["auction_hist_start"], p["auction_hist_end"]),
            ("realised", "X_realised", p["realised_hist_start"], p["realised_hist_end"]),
            ("ahead", "X_ahead", p["target_start"], p["target_end"]),
            ("calendar", "X_calendar", p["target_start"], p["target_end"]),
        ]:
            want = df.loc[pd.Timestamp(start):pd.Timestamp(end), cols[block]].to_numpy(np.float32)
            got = z[key][i]
            check(f"{block} values match source",
                  want.shape == got.shape and np.allclose(want, got, equal_nan=True, atol=1e-3),
                  f"origin {p['origin_day']} block {block}")
        want_y = df.loc[pd.Timestamp(p["target_start"]):pd.Timestamp(p["target_end"]),
                        cols["target"]].to_numpy(np.float32)
        check("target values match source",
              np.allclose(want_y, z["Y"][i], equal_nan=True, atol=1e-3),
              f"origin {p['origin_day']}")

    # ------------------------------------------------------------- 4. splits sane
    origins = pd.to_datetime(man["origin_days"])
    tr, va, te = z["idx_train"], z["idx_val"], z["idx_test"]
    check("splits disjoint", len(set(tr) & set(va) & set(te)) == 0)
    if len(tr) and len(va):
        check("train precedes val", origins[tr].max() < origins[va].min())
    if len(va) and len(te):
        check("val precedes test", origins[va].max() < origins[te].min())

    # ----------------------------------------------------------------- report
    print(f"{CHECKS} checks run over {len(prov)} origins "
          f"({len(picks)} value-level spot checks)")
    if FAILS:
        print(f"\nFAILED ({len(FAILS)}):")
        for f in FAILS[:20]:
            print("  -", f)
        sys.exit(1)
    print("PASS -- no feature is visible before it is published")


if __name__ == "__main__":
    main()
