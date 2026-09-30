"""
Gate-closure audit: the single source of truth for what each column is and when
it becomes known.

THE RULE
--------
A forecast origin is 12:00 CET on day D (SDAC day-ahead order book closure).
The target is delivery day D+1, hours 00:00-23:00 local.

A value may enter the feature matrix only if it was published before 12:00 CET
on day D. Publication time depends on the variable, not just on its timestamp,
so columns are grouped into availability classes and each class gets its own
cutoff.

Timings confirmed from the SDAC / MRC day-ahead process schedule:
    09:30-10:30 CET   cross-zonal capacities published
    12:00 CET         order book gate closure          <-- our origin
    12:45 CET         preliminary coupling results
    12:57 CET         final coupling results
Gate closure is 12:00 CET uniformly across ALL SDAC coupled zones, so there are
no earlier-closing neighbours among DE/FR/PL whose prices could be exploited.

CLASSES
-------
AUCTION_PUBLISHED
    Outputs of the day-ahead coupling: prices, spreads, scheduled exchanges, net
    positions. Published at ~12:45 CET on day D-1 for the whole of delivery day
    D. Therefore at our origin (12:00 on D) the ENTIRE of day D is already known.
    Cutoff: end of day D.

    NOTE: project 1 dropped sched_exch_* and net_pos_* entirely in --pre-auction
    mode. That is conservative but throws away real information: yesterday's
    scheduled exchanges are public. Here they are admitted as HISTORY only. Using
    them for delivery day D+1 would be leakage; using them for day D is not.

REALISED
    Metered outturn: actual load, actual wind/solar, physical flows. Published
    with roughly an hour's lag. Conservatively cut at 09:00 CET on day D, which
    leaves a 3-hour safety margin before gate closure.
    Cutoff: D 09:00.

FORECAST_AHEAD
    TSO day-ahead forecasts and outage notices for delivery day D+1.
    Cutoff: available for all of D+1. These are the only non-calendar features
    that see the target day.
    Official deadlines (Regulation (EU) 543/2013): the day-ahead load forecast is
    due two hours before gate closure (Art. 6(1)(b)); the scheduled generation and
    wind/solar forecasts are due by 18:00 Brussels time on day D (Art. 14(1)(c),
    (d)), i.e. after gate closure. They are used here on the assumption that they
    are published before 12:00 on day D (PAPER.md, Section 3.2). The derived
    residual-load forecasts inherit this assumption.

CALENDAR
    Deterministic. Available for any date.

EXCLUDED_BROKEN
    Dropped for coverage reasons, not leakage: approximately 100% missing from
    2024 onward, or more than 20% missing overall.
"""

TARGETS = ["spread_DE_FR", "spread_DE_PL"]

# Dropped: ~100% missing from 2024 onward, or >20% missing overall.
EXCLUDED_BROKEN = [
    "wind_DE",
    "solar_DE",
    "renewable_act_DE",
    "sched_exch_DE_BE",
    "sched_exch_DE_NO_2",
]

# Known for the whole of day D at a 12:00-on-D origin (published 12:45 on D-1).
AUCTION_PUBLISHED = [
    "price_DE", "price_FR", "price_PL",
    "spread_DE_FR", "spread_DE_PL",
    "sched_exch_DE_FR", "sched_exch_DE_NL", "sched_exch_DE_AT",
    "sched_exch_DE_CH", "sched_exch_DE_PL", "sched_exch_DE_CZ",
    "sched_exch_DE_DK_1", "sched_exch_DE_DK_2", "sched_exch_DE_SE_4",
    "net_pos_FR", "net_pos_PL",
]

# Metered outturn, ~1h publication lag. Cut at 09:00 on day D.
REALISED = [
    "load_act_DE", "load_act_FR", "load_act_PL",
    "wind_FR", "solar_FR", "renewable_act_FR",
    "wind_PL", "solar_PL", "renewable_act_PL",
    "flow_DE_FR", "flow_DE_PL",
]

# TSO forecasts for delivery day D+1 (see the FORECAST_AHEAD note on deadlines).
FORECAST_AHEAD = [
    "load_fc_DE", "gen_fc_DE", "wind_solar_fc_DE",
    "load_fc_FR", "gen_fc_FR", "wind_solar_fc_FR",
    "load_fc_PL", "gen_fc_PL", "wind_solar_fc_PL",
    "outage_DE", "outage_FR", "outage_PL",
]

CALENDAR = [
    "hour", "dayofweek", "month", "is_weekend",
    "is_holiday_DE", "is_holiday_FR", "is_holiday_PL",
]

# Columns computed in build_dataset.add_derived(). Declared HERE, statically, so
# the audit has one source of truth: a derived feature that is not declared in
# this file cannot enter the dataset. (An earlier version registered these at
# build time by mutating CLASS_OF, which meant the leakage test -- importing this
# module fresh -- could not see them. The test caught it.)
DERIVED_AUCTION = [
    "daymin_price_DE", "daymax_price_DE",
    "daymin_price_FR", "daymax_price_FR",
    "daymin_price_PL", "daymax_price_PL",
]
DERIVED_FORECAST = [
    "resid_load_fc_DE", "resid_load_fc_FR", "resid_load_fc_PL",
]
AUCTION_PUBLISHED = AUCTION_PUBLISHED + DERIVED_AUCTION
FORECAST_AHEAD = FORECAST_AHEAD + DERIVED_FORECAST

# Per-class cutoff, expressed as (day offset from origin day D, last hour inclusive).
# None means "no cutoff -- available for the target day D+1".
CUTOFFS = {
    "AUCTION_PUBLISHED": (0, 23),   # through D 23:00
    "REALISED": (0, 9),             # through D 09:00
    "FORECAST_AHEAD": None,         # all of D+1
    "CALENDAR": None,               # deterministic
}

CLASS_OF = {}
for _name, _cols in [
    ("AUCTION_PUBLISHED", AUCTION_PUBLISHED),
    ("REALISED", REALISED),
    ("FORECAST_AHEAD", FORECAST_AHEAD),
    ("CALENDAR", CALENDAR),
]:
    for _c in _cols:
        CLASS_OF[_c] = _name

HISTORY_CLASSES = ("AUCTION_PUBLISHED", "REALISED")
AHEAD_CLASSES = ("FORECAST_AHEAD", "CALENDAR")

def classify(columns):
    """Split an iterable of column names into {class: [cols]} plus 'UNKNOWN'.

    Any column landing in UNKNOWN is a bug: every column must be explicitly
    audited before it can be used. Fail rather than guess.
    """
    out = {k: [] for k in
           ["AUCTION_PUBLISHED", "REALISED", "FORECAST_AHEAD",
            "CALENDAR", "EXCLUDED_BROKEN", "UNKNOWN"]}
    for c in columns:
        if c in EXCLUDED_BROKEN:
            out["EXCLUDED_BROKEN"].append(c)
        elif c in CLASS_OF:
            out[CLASS_OF[c]].append(c)
        else:
            out["UNKNOWN"].append(c)
    return out

def audit(columns):
    """Raise if any column is unaudited. Call this before building anything."""
    groups = classify(columns)
    if groups["UNKNOWN"]:
        raise ValueError(
            "Unaudited columns -- classify them in feature_classes.py before use: "
            + ", ".join(sorted(groups["UNKNOWN"]))
        )
    return groups
