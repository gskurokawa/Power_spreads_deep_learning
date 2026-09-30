# Results

> **Superseded for the benchmark comparison.** The comparisons against LEAR and the
> seasonal naive forecast below use fixed residual offsets to give those models a
> predictive distribution. With quantile regression post-processing
> (`experiments/lear_fix.py`), LEAR attains pinball loss 6.45 and is jointly best
> with `mlp_pool` (6.80); see `PAPER.md` Sections 6.1–6.5. Results for the neural
> models below are unaffected.

All numbers regenerate from `experiments/results.csv` and `experiments/daily/` via
`python experiments/analyse.py`. Walk-forward, 23 quarterly retrains, expanding
window, 1,988 out-of-sample forecast days, seeds 0-2 (the set every model
completed).

---

## 0. What was run, and why

### The forecasting task

One forecast per day. At **12:00 CET** — the moment the European day-ahead auction
closes — predict all **24 hourly values** of two spreads for tomorrow: **DE-FR** and
**DE-PL**. Not a point estimate but **seven quantiles** (5, 10, 25, 50, 75, 90, 95%),
because the target has a point mass at exactly zero and a point forecast under MAE
simply learns to hug zero. How large that mass is depends entirely on the window:

| Exact zeros | DE-FR | DE-PL |
|---|---|---|
| Full sample, 2019-2026 | 30.3% | 7.1% |
| Walk-forward test, 2021-2026 | 25.0% | 9.4% |
| Fixed-split test, 2025-2026 | **8.5%** | 5.6% |

The DE-FR point mass has collapsed from 30% to 8.5% — more interconnection, more
hours where the border binds. That is the regime change this project's protocol
exists to handle, and it is also a warning: a single figure for "how zero-inflated
is this target" is meaningless without the window attached.

That framing is deliberate and costly. The auction clears once daily for 24 hours
at once, so **one day is one observation** — 2,705 of them, not the 66,432 hours in
the source data. Using overlapping hourly windows would have claimed 24× the
sample from 96% redundant rows. Everything downstream follows from accepting
~1,800 training days instead.

Every feature is **gate-closure audited**: classified by its official ENTSO-E
publication time and admitted only if it was public before 12:00 on the forecast
day. Auction results for day D publish at 12:45 on D-1, so all of day D is known;
metered outturn lags an hour and is cut at 09:00; TSO forecasts for D+1 are the
only non-calendar features allowed to see the target day. 19,194 assertions
enforce this.

### The protocol

**Walk-forward, not a single split.** 23 quarterly retrain points from 2021 to
2026, each model tested on the following quarter — **1,988 out-of-sample forecast
days** against 562 on a fixed split. Chosen because DE-FR zero-spread hours fell
from 49% in 2021 to 6% in 2026: one split would have measured one regime. It also
earned its cost directly, surfacing two bugs a fixed split hid (LEAR's feature
matrix outgrowing a rolling window, and an `asinh` transform exploding to 10¹³ MAE
on the zero-inflated target).

**Two window policies**, run as a matched pair: *expanding* (keep everything) and
*rolling 2-year* (discard old data). This is the drift experiment — it measures
whether old data hurts rather than assuming it.

**Multiple seeds** per configuration, because the between-architecture gaps turned
out comparable to the within-architecture noise. Measuring that floor (0.59 MAE on
DE-FR) is what stopped several apparent findings from being written up.

**3,581 fits, 31.9 hours of compute**, covering the eight further models listed
below as well as the five reported here.

### The models

| Model | What it is | Why it is in the study | Params |
|---|---|---|---|
| `seasonal_naive` | Same hour previous day (Tue-Fri) or previous week (Sat-Mon) | The floor. If a model cannot beat this, nothing else matters | 0 |
| `lear` | Lasso Estimated AutoRegressive: one BIC-selected Lasso per target hour, asinh-transformed | **The benchmark that counts.** Lago et al.'s standard EPF baseline; "we beat LEAR" is a claim a reader can evaluate | — |
| `mlp_pool` | 2×64 MLP; history compressed to mean/std/min/max/last per feature | **No sequence structure at all.** The control against which recurrence and attention are measured | 43,936 |
| `lstm` | Encoder-decoder; one LSTM per history stream, decoder over the 24 target hours | **Recurrence.** Reads all 168 hours in order. Controls for "does recurrence alone explain whatever attention gains?" | 48,942 |
| `transformer` | Hand-written attention over 38 tokens: 7 auction-days + 7 realised-days + 24 target-hours | **The subject of the project.** Attention written from scratch and verified against `nn.MultiheadAttention` on identical weights | 79,550 |

Eight further models were run and are **not reported here**, to keep the
comparison readable. All of their rows remain in `experiments/results.csv` and
`experiments/daily/`: four `*_skip` variants (each architecture plus a rank-8
linear path from the inputs to the output, per El Mahtouta & Ziel 2026),
`mlp_flat` (history flattened rather than pooled, the flattening ablation),
`linear` (ridge on all 6,072 features — superseded as a linear reference by LEAR,
which is both stronger and externally recognisable), `mlp_sklearn` (an untuned
squared-loss MLP, the weak-effort floor), and `hurdle_lstm` (a two-headed model
for the zero point mass). §5 carries forward the findings from that set that
matter.

Three design constraints hold across all of them, and they are what make the
comparison about mechanism rather than plumbing:

1. **Identical inputs, loss, splits and scoring.** Every model sees the same
   gate-closure-audited features and is judged by the same pinball loss.
2. **Identical output head.** The LSTM and transformer both emit through a
   686-parameter head (`Linear(48, 14)`) applied to each of the 24 target-hour
   states with shared weights. The alternative -- concatenating the 24 states into
   one 1,152-wide row and mapping it to all 336 outputs at once -- costs
   1,152 x 336 + 336 = **387,408 parameters**, nearly five times the transformer's
   entire budget, and would let each hour learn its own private output rule
   instead of forcing all 24 through one shared mapping.
3. **Parameter parity within 1.8×** for the three compared architectures:
   `mlp_pool` 43.9k, `lstm` 48.9k, `transformer` 79.6k. No architecture wins by
   being larger.

### How they are scored

**Pinball loss** is primary — it is the training objective and the only metric
that sees the whole predictive distribution. **MAE on the conditional median** is
reported alongside for comparability with point-forecast benchmarks. Both are
**split by congestion state**, because a model can score well on a target with a
large zero mass simply by predicting near zero everywhere. Note that this split is
computed from the **realised** outcome (`y_true == 0`), so it is a diagnostic of
where a model's error sits, not a performance figure that could have been known in
advance.

Then a four-step battery, on daily losses aggregated to one value per forecast
origin (testing on 47,712 hourly errors when there are 1,988 independent forecasts
would overstate the sample 24-fold):

- **Diebold-Mariano**, Newey-West HAC, Harvey-Leybourne-Newbold corrected
- **Giacomini-Rossi fluctuation test** — is the ranking *stable*? Critical values
  simulated from the limiting Brownian functional rather than copied
- **Model Confidence Set** — with 10 pairwise comparisons among the five reported
  models, which are indistinguishable from the best, in one procedure
- **Seed distributions**, to establish the floor any difference must clear

---

## 1. The headline: the metrics disagree, and so do the two spreads

Walk-forward, expanding window, 23 quarters, seeds 0-2 (the set every model
covers). Pinball is computed over **both** spreads; the two MAE columns are
per-spread. Noise floors: 0.59 on DE-FR, 0.69 on DE-PL.

| Model | Pinball (both) | DE-FR MAE | rank | DE-PL MAE | rank |
|---|---|---|---|---|---|
| **mlp_pool** | **6.80** | 20.02 | 4 | **22.21** | **1** |
| **transformer** | 8.21 | **19.08** | **1** | 24.83 | 2 |
| LEAR | 8.40 | 19.52 | 2 | **28.90** | **4** |
| lstm | 8.50 | 19.70 | 3 | 25.02 | 3 |
| seasonal naive | 9.62 | 25.12 | 5 | 29.69 | 5 |

Three disagreements, and each one would have produced a different paper.

**Point forecast vs distribution.** The transformer has the best DE-FR point
forecast and the third-worst distribution. The pooled MLP has the best
distribution and a middling point forecast. A project reporting only MAE would
have concluded "attention wins"; one reporting only pinball, "attention is
near-useless". Both defensible, both half the story.

**DE-FR vs DE-PL.** The ranking barely survives the change of target. LEAR goes
from 2nd to 4th of 5, ahead of nothing but the naive floor. `mlp_pool` goes from
4th to 1st.

**And that reframes the project's main claim.** On DE-FR the field is a tie and
LEAR is in it. On DE-PL, LEAR scores 28.90 against `mlp_pool`'s 22.21 — a gap of
**6.69 against a 0.69 noise floor, nearly ten times over**. Every neural model
beats it. So "the neural models match the linear benchmark but do not beat it" is
a DE-FR statement, not a general one. Stated for both targets:

> On DE-FR everything ties. On DE-PL the linear benchmark fails and the pooled
> MLPs win decisively.

**Why the two targets differ, most likely.** DE-FR is 25% exact zeros in the test
period; DE-PL is 9%. LEAR's design — a per-hour Lasso on an asinh-transformed
target — suits a heavy point mass and a long stable coupling history. DE-PL has
neither: less of a point mass, and Polish market coupling arrived partway through
the sample. This is a hypothesis; it is testable from the stored forecasts in
`experiments/daily/` without refitting anything, and has not been tested.

## 2. Model Confidence Set: the pooled MLP, and nothing else

Hansen-Lunde-Nason at 90%, moving-block bootstrap, daily pinball losses,
T = 1,988, seed 0. Only **`mlp_pool`** survives; every other model is eliminated.

| Eliminated, in order | p | Daily pinball |
|---|---|---|
| seasonal_naive | 0.000 | 9.761 |
| lstm | 0.002 | 8.833 |
| transformer | 0.002 | 8.549 |
| LEAR | 0.006 | 8.611 |
| **SURVIVOR: mlp_pool** | | **6.801** |

One qualification: the MCS ran on seed 0 alone. The gap from `mlp_pool` to the
next model is 1.0 against seed ranges of roughly 0.15, so the separation is
comfortable, but a single-seed set-level guarantee is weaker than the procedure's
usual reading.

**Defensible statement:** the pooled MLP is best on the distributional metric and
clearly separated from everything else tested.

## 3. The research question, answered: attention adds nothing

The project asked whether attention buys anything over simpler architectures on
the same inputs, at comparable size.

| | Pinball | Parameters |
|---|---|---|
| `mlp_pool` | **6.80** | 43,936 |
| `transformer` | 8.21 | 79,550 |
| `lstm` | 8.50 | 48,942 |

Attention is **21% worse** on the distributional metric than a two-layer
feedforward network that has no notion of sequence at all — with 1.8× the
parameters. Recurrence is worse still. The ordering is
MLP → transformer → LSTM: no relationship to architectural sophistication.

On DE-FR MAE alone the order reverses (transformer 19.08, `mlp_pool` 20.02), but
that gap is inside the 0.59 noise floor while the pinball gap is not. Where the
two metrics conflict and only one of the gaps clears noise, the pinball reading
is the one that stands.

## 4. Relative performance is not stable — no pooled number is the right summary

Giacomini-Rossi fluctuation test, μ = 0.3, 5% critical value 3.06 (simulated from
the limiting functional, not copied from the table):

| Pair | Statistic | Verdict | Sign changes | Peak |
|---|---|---|---|---|
| transformer vs mlp_pool | 12.15 | UNSTABLE | 2 | 2022-05 |
| mlp_pool vs LEAR | 8.62 | UNSTABLE | 2 | 2022-08 |
| lstm vs LEAR | 4.74 | UNSTABLE | 2 | 2023-01 |
| transformer vs LEAR | 4.73 | UNSTABLE | 2 | 2023-01 |
| transformer vs lstm | 3.72 | UNSTABLE | 2 | 2021-11 |

All five reject stability, and every one changes rank twice over the sample. So
no single pooled number — not the DM statistic, not the mean in §1 — is the right
summary of any of these pairs. Whichever model you name as better, there are
stretches of 1,988 days where the other one was.

Every peak falls in 2022-23. That is the energy crisis and the collapse in
DE-FR price convergence (zero-spread hours fell from 49% in 2021 to 6% by 2026),
and it is when the models most disagree.

## 5. What was larger than noise, and what wasn't

Seed noise floors, median within-quarter standard deviation across seeds:
**0.59 MAE on DE-FR, 0.69 on DE-PL** (seeds 0-2, the set every model covers;
0.53 / 0.63 using all eight seeds of the models that have them).

Real:
- Expanding beats rolling for **every** model, +0.17 to +1.37, rolling winning
  22-48% of quarters. Old data does not hurt despite the regime change — the
  reverse of the prior this project started with.
- **The pooled MLP's pinball advantage over everything else.** 6.80 against 8.21
  for the transformer, 8.40 for LEAR and 8.50 for the LSTM; sole survivor of the
  Model Confidence Set, every rival eliminated at p ≤ 0.006.
- **LEAR's failure on DE-PL.** 28.90 against `mlp_pool`'s 22.21, a gap of 6.69
  against a 0.69 noise floor. The largest single effect in the study, and the only
  place any architecture clearly beats the benchmark.
- **Pooling beats flattening**, from the variants held back from this report:
  `mlp_flat` 21.56 DE-FR MAE against `mlp_pool`'s 20.02, with 4.8× the parameters.
  Five summary statistics per feature beat 168 raw timesteps. The gap is widest in
  2022 (4.74) and near zero in calm 2024, which points at the nonlinear summaries —
  std, min, max — rather than at parameter count alone.
- **The linear skip helps the sequence models and not the MLPs**, also from the
  held-back set: `lstm` pinball −0.92, `transformer` −0.47, both MLPs unchanged.
  It changes no ranking here — a skip-equipped LSTM reaches 7.58, still short of
  `mlp_pool`'s 6.80.

Not real:
- The top of the **DE-FR** MAE table. Transformer 19.08, LEAR 19.52 and `lstm`
  19.70 all sit within one noise floor.
- Any ranking claim stated without naming which spread it refers to — see §1.

## 6. Caveats a reader will raise

**No hyperparameter tuning.** Every neural model ran on one hand-chosen
configuration, sized for parameter parity, never searched. LEAR re-selects its
Lasso penalty by BIC on every one of its 48 regressions, at every one of the 23
retrains. So the benchmark adapts to each new window and the neural models do not
— a handicap working against this study's own models, and a live alternative
explanation for why they only tie with LEAR on DE-FR. Lago et al. run 1,500
hyperopt evaluations; this study ran zero.

**The transformer is below published minimums.** `d_model=48` against a floor of
64 in the one published EPF transformer paper, 2 layers against 4-6, `d_ff=96`
against 256. Defensible on 628-2,586 training days depending on the quarter, but
it means "attention doesn't help here" and "this transformer was too small" are
not separated. The most direct test — shorter history tokens (12h or 6h instead of
24h), so the model can see within-day shape — has not been run.

**DM, MCS and fluctuation tests ran on seed 0.** T = 1,988 days gives them power,
but they do not see seed variation.

**Three seeds for the transformer**, eight for the rest. Every table here is
therefore computed on seeds 0-2, discarding five seeds of the models that have
them. Taking the transformer up to eight would let the whole report use all of
them.

**Eight models were run and are not reported here**, and remain in
`experiments/results.csv` and `experiments/daily/`. The consequence that matters:
§3 uses the no-skip versions of the sequence models, not their best versions. A
skip-equipped LSTM reaches 7.58 pinball rather than 8.50 — still short of
`mlp_pool`, so the conclusion holds, but by a narrower margin than §3 shows.
