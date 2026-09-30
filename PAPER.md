# A Comparison of Attention, Recurrence and Linear Models in Forecasting European Power Price Spreads

**Glen Kurokawa**
September 2026

*Working paper. Every table is reproduced by `python experiments/paper_tables.py`
from the files in `experiments/` (Appendix A).*

---

## Abstract

This study forecasts the complete 24-hour profile of two European cross-zonal
day-ahead price spreads, Germany–France (DE–FR) and Germany–Poland (DE–PL), as
seven-quantile predictive distributions issued daily at 12:00 Central European Time,
the gate closure of the day-ahead auction. Inputs are restricted to information
expected to be public before the auction closes (Section 3.2). Five models are
compared: a naive forecast; LEAR, the standard linear benchmark in electricity price
forecasting; a multilayer perceptron (MLP) operating on pooled summary statistics of
the input history; a long short-term memory (LSTM) network; and a transformer. The
three deep learning models are matched in parameter count to within a factor of 1.8.
Evaluation follows a walk-forward design.

Two results are reported. First, LEAR and the pooled MLP are jointly best. Second,
attention and recurrence do not improve accuracy: the transformer and the LSTM are
significantly worse. Both results differ from most earlier comparisons on day-ahead
price levels, in which transformers match or outperform LEAR-type and MLP models (Wang
et al., 2024; Elashhab et al., 2026) and recurrent networks outperform a LEAR-type
model (Lago et al., 2018). 

---

## 1. Introduction

### 1.1 Cross-zonal power price spreads

A cross-zonal price spread is the difference between the day-ahead clearing prices of
two bidding zones. A bidding zone is a geographic area within which electricity is
traded at a single uniform price; the three zones used in this study are DE
(Germany–Luxembourg), FR (France) and PL (Poland). The European day-ahead market is
cleared by market coupling: an algorithm clears the day-ahead auctions of all
participating zones simultaneously, subject to the capacity of the interconnectors
between them, and produces one price per zone per hour. Its pan-European
implementation is the Single Day-Ahead Coupling (SDAC) mechanism. The SDAC order book
closes at 12:00 Central European Time on the day before delivery; this instant, after
which no further bids may be submitted, is termed gate closure. Preliminary results
are published at 12:45 and final results at 12:57. When the interconnector between two
zones is not congested, the coupling algorithm equalises their clearing prices and the
spread is exactly zero. When the interconnector binds, that is, when its full capacity
is in use, the prices separate. 

### 1.2 Research question

This study examines how well deep learning models compare in
power price spread forecast accuracy.

The deep learning models compared are a transformer, a long short-term memory (LSTM)
network and a multilayer perceptron (MLP). The transformer in this study represents its inputs as 38 tokens, the input elements that it compares with one another: one for each of the seven days of auction history, one for each of the seven days of realised history, and one for each of the 24 delivery hours. The tokens are processed by two encoder layers, each with four attention heads. The LSTM network in this study has an encoder–decoder structure, a standard
arrangement for sequence forecasting: the encoder reads the input history and
summarises it in its final state, and the decoder starts from that state and produces
one output for each of the 24 delivery hours. The MLP in this study has
two hidden layers of 64 units and receives each input variable's history as five
summary statistics; it is referred to as the pooled MLP. Their inputs are described in
Section 4.3.

The three deep learning models are compared under controlled conditions: identical
inputs, training objective, data partitions, re-estimation schedule, output layer and
scoring, with parameter counts matched to within a factor of 1.8. Re-estimation means
fitting a model afresh on the data available up to a given date. In the principal
results every model is re-estimated at the start of each calendar quarter and used
unchanged for that quarter.

---

## 2. Literature

### 2.1 The standard benchmark

Lago, Marcjasz, De Schutter and Weron (2021) constitutes the reference point for
electricity price forecasting. It specifies five open datasets, a standard two-year
test period, and two benchmark models: LEAR, a Lasso-estimated autoregressive
specification fitted separately for each delivery hour, and an ensemble of deep neural
networks. The implementation is distributed as the `epftoolbox` package. In the
published benchmark, LEAR is re-estimated every day on each of four calibration
windows (8 weeks, 12 weeks, 3 years and 4 years), and the four forecasts are averaged.

### 2.2 Predictive distributions from point forecasts

LEAR is a point forecaster: it produces a point forecast, a single predicted value for
each delivery hour. The models in this study are evaluated instead on quantile
forecasts. For probability level q, a quantile forecast is the value below which the
realised outcome is predicted to fall with probability q. Seven levels are forecast:
0.05, 0.10, 0.25, 0.50, 0.75, 0.90 and 0.95. The set of seven quantile forecasts
issued for one delivery hour and one spread is termed a predictive distribution.

The construction of a predictive distribution from a model's point forecast, by a
separate statistical procedure applied after the model has been estimated, is termed
post-processing. The electricity price forecasting literature has developed several
such methods, the most established being Quantile Regression Averaging (Nowotarski and
Weron, 2015), in which quantile regressions are estimated on the point forecasts of
one or more models. The present study applies the single-model form of this method to both
benchmark models and refers to it as quantile regression post-processing: for each
quantile level, the realised value is regressed on functions of the point forecast by
linear quantile regression, estimated on past out-of-sample forecasts, so that the
width of the predictive distribution varies with the point forecast (Section 4.4).

### 2.3 Architecture comparisons in day-ahead electricity price forecasting

Lago, De Ridder and De Schutter (2018) compare forecasting methods on Belgian
day-ahead prices, re-estimated daily. A feedforward network with two hidden layers is significantly more accurate than recurrent networks including LSTM. Recurrent networks are in turn significantly more accurate than a model similar to LEAR. No transformer is included.

Wang, Liu, Zhang and Wang (2024) forecast day-ahead prices in China and New England, re-estimating daily. A Lasso-estimated linear model of the LEAR type and the feedforward network of Lago et al. (2018) have
higher error than most of the transformers evaluated. A linear model is the most accurate in both markets. The study evaluates point forecasts only. 

Elashhab, Papineni, Dorn, Hagenmeyer and Schäfer (2026) compare six deep learning
models for day-ahead prices in the Germany–Luxembourg bidding zone. Their benchmarks are simple
rules such as repeating the previous day's price; LEAR is not included. A standard transformer and an LSTM
network are the least accurate. The study reports results from point forecasts
only.

In these studies, MLPs outperform recurrent networks like LSTMs, but recurrent networks outperform a LEAR-type model (Lago et al., 2018), and LEAR and MLPs do not outperform transformers (Wang et
al., 2024; Elashhab et al., 2026). None of the three evaluates probabilistic
forecasts, and only Lago et al. (2018) reports significance tests.

### 2.4 Positioning of this study

Each component of this study has precedent: the benchmark (Section 2.1), the
post-processing method (2.2) and the architectures (2.3). The study applies them to
European cross-zonal spreads, with probabilistic evaluation and significance tests.
Its principal finding, that LEAR and a multilayer perceptron are significantly more
accurate than a transformer and an LSTM network, agrees with the earlier comparisons
only in that a feedforward network outperforms a recurrent network (Lago et al.,
2018). In its other parts it differs from them. Possible explanations are discussed in Section 7.1.

---

## 3. Data and input availability

### 3.1 The forecasting task

One forecast is issued per day. The instant at which a forecast is issued is termed
the forecast origin, and the information used in the forecast is restricted to what is
expected to be public before it (Section 3.2). The forecast origin is 12:00 Central
European Time on day D and coincides with gate closure. At that instant the model
predicts all 24 hourly values of two spreads, DE–FR and DE–PL, for the delivery day:
the day for which prices are forecast, denoted D+1. The output is a set of seven
quantile forecasts per hour per spread, giving 24 × 2 × 7 = 336 predicted values per
forecast origin.

The unit of observation is the forecast origin rather than the delivery hour. The
auction clears once daily for all 24 hours jointly, so the 24 hourly values of a
single delivery day are produced by one clearing event and are forecast from the
same inputs. This yields 2,705 observations from 66,432 source hours. Constructing
overlapping hourly windows would have produced 24 times as many rows, of which 96%
of the content would be shared with a neighbouring row, and would have understated
every standard error accordingly. Model estimation therefore uses between 628 and
2,586 training observations, depending on the re-estimation date.

### 3.2 Input data and publication times

The input data fall into four groups according to when they are published. For each
group, the table gives the latest time up to which its values are used when a forecast
is issued at 12:00 on day D. Every column of the source data is assigned to one of
these groups; if a column is not assigned, construction of the dataset stops with an
error.

| Input data | Columns | Values used up to | Reason |
|---|---|---|---|
| Day-ahead auction results | 22 | end of day D | Auction results for day D are published at 12:45 on day D−1, so the whole of day D is public at 12:00 on day D |
| Realised outcomes (measured load, generation and flows) | 11 | 09:00 on day D | Measured values are published approximately one hour after the event; using values only up to 09:00 leaves three hours of margin |
| Day-ahead forecasts by transmission system operators | 15 | all of day D+1 | Load forecasts: official deadline two hours before the auction closes. Wind, solar and generation forecasts: official deadline 18:00 on day D; used on the assumption that they are published before 12:00 |
| Calendar variables | 7 | any date | Known in advance |

Five columns were excluded because of missing data rather than publication timing:
German actual wind, solar and aggregate renewable generation, which are
approximately 100% missing from 2024 onward, and two scheduled-exchange series.

Two consequences merit explicit statement.

First, the auction history window extends eleven hours beyond the forecast origin.
At 12:00 on day D, the model receives the clearing price for hour 23:00 of day D.
This is admissible because that value was published at 12:45 on day D−1.

Second, the auction and realised history windows are consequently not aligned in
time. Both contain 168 hourly observations, but the auction window ends at 23:00 on
day D and the realised window at 09:00 on day D, an offset of exactly 14 hours,
verified to hold in all 2,705 observations. The two series are therefore never
concatenated along the feature axis in any architecture.

### 3.3 The zero probability mass

| Proportion of hours exactly zero | DE–FR | DE–PL |
|---|---|---|
| Full sample, 2019–2026 | 30.3% | 7.1% |
| Walk-forward test period, 2021–2026 | 25.0% | 9.4% |
| Fixed-split test period, 2025–2026 | 8.5% | 5.6% |

The DE–FR mass declined from 30.3% over the full sample to 8.5% over the most recent
period. This non-stationarity motivates the walk-forward protocol and establishes
that any single zero-inflation statistic must be reported with its window.

---

## 4. Models

| Model | Specification | Function in the study | Parameters |
|---|---|---|---|
| `naive` | Copies the spread at the same hour on an earlier day (Section 4.2) | Lower reference | 0 |
| `lear` | Lasso Estimated AutoRegressive: one Lasso regression per delivery hour and spread, penalty selected by the Bayesian Information Criterion, spread transformed by the inverse hyperbolic sine | Standard benchmark (Lago et al., 2021) | selected per fit |
| `mlp_pool` | Multilayer perceptron, two hidden layers of 64 units; each history variable summarised by its mean, standard deviation, minimum, maximum and most recent value | Control encoding no sequential structure | 43,936 |
| `lstm` | Encoder: one LSTM per history stream reads the 168 hours in order. Decoder: a further LSTM steps through the 24 delivery hours | Recurrence | 48,942 |
| `transformer` | Self-attention over 38 tokens: 7 auction-day, 7 realised-day and 24 delivery-hour | Attention | 79,550 |

### 4.1 What is shared and what differs

**Shared by all five models:** the data, the training windows, the quarterly
re-estimation schedule, the 1,988 test days and the scoring procedure.

**Shared by the three deep learning models, and not by the benchmarks:**

- *Inputs.* All four input blocks: auction history (168 hours × 22 variables),
  realised history (168 × 11), day-ahead forecasts for the delivery day (24 × 15) and
  calendar variables (24 × 7).
- *Training objective.* Pinball loss, which penalises each quantile forecast in
  proportion to its distance from the realised value, weighting errors above and below
  the forecast by q and 1 − q respectively (Section 5.2), so that the seven quantile
  forecasts are estimated directly.
- *Estimation.* Adam optimiser, learning rate 0.001, batch size 64, weight decay
  0.0001, at most 300 passes over the training data, stopping early when the loss on
  the final 90 days of the training window has not improved for 25 passes. One
  configuration per architecture, chosen by hand and not searched.
- *Output layer* (recurrent network and transformer). A 686-parameter linear layer
  mapping each delivery hour's 48-dimensional state to 14 values (2 spreads × 7
  quantiles), with the same weights for all 24 hours.
- *Size.* Parameter counts within a factor of 1.8 of one another.

**LEAR differs by definition.** Its inputs are its standard autoregressive design:
the two spreads for days D, D−1, D−2 and D−6; the three zone prices for day D; the
daily minimum and maximum price per zone; daily means of the day-ahead forecasts;
calendar variables; and the 15 day-ahead forecasts at the delivery hour — 307
columns in total, and no realised-outturn data. It is estimated by least squares
with a Lasso penalty on a transformed spread, not on pinball loss, and it selects
its own penalty at every re-estimation. It produces a point forecast, which is
converted into a predictive distribution by post-processing (Section 4.4).

**The naive forecast** uses only past values of the two spreads (Section 4.2) and is
converted into a predictive distribution in the same way.

### 4.2 Construction of the naive forecast

The naive forecast is a point forecast that copies an observed value of the spread
from an earlier day and estimates no parameters. For each delivery hour, it is the
realised spread at the same hour on an earlier day, chosen according to the day of the
week of the delivery day:

| Delivery day | Value copied |
|---|---|
| Tuesday, Wednesday, Thursday, Friday | Same hour on day D, the day on which the forecast is issued |
| Saturday, Sunday, Monday | Same hour on the same weekday one week before the delivery day |

The distinction reflects the difference between working days and weekends: the
spread on a working day resembles that of the preceding working day, whereas the
spread on a Saturday, Sunday or Monday resembles that of the same day of the previous
week more closely than that of the preceding day. The rule follows the convention
used for naive benchmarks in electricity price forecasting.

All values used are available at the forecast origin: the spreads for every hour of
day D were published at 12:45 on day D−1. The code identifier for this model is
`seasonal_naive`.

### 4.3 Input representation in the deep learning models

All values are standardised to zero mean and unit variance using statistics computed
on the training window alone. The architectures differ only in how the 168 hours of
history are processed:

- `mlp_pool` reduces each variable's 168 values to five summary statistics.
- `lstm` processes the 168 hours in temporal order, maintaining a recurrent state.
- `transformer` groups the 168 hours into seven day-level tokens, each formed by a
  linear projection of that day's 24 hours × 22 variables to 48 dimensions. Each of
  the 24 delivery hours forms its own token from that hour's day-ahead forecasts and
  calendar variables. The 38 tokens are processed by two pre-normalisation encoder
  layers with four attention heads each. Because each history day is projected to a
  single token, the model cannot distinguish one hour of a history day from another.

Quantile crossing is prevented by construction: each network predicts the lowest
quantile and six non-negative increments, obtained by the softplus function, which
are accumulated.

The multi-head self-attention mechanism is implemented directly. A test copies its
weights into `torch.nn.MultiheadAttention` and asserts numerical agreement of the
outputs.

### 4.4 Predictive distributions for the point-forecast models

For each spread j and each quantile level q, a linear quantile regression is
estimated of the form

    y = Σ_h α_{q,h} · 1[hour = h]  +  β_q · f  +  γ_q · |f|

where y is the realised spread, f is the model's point forecast and 1[hour = h] are
24 hour indicators. The term in |f| allows the interval to widen as the magnitude of
the point forecast increases, in either direction. The regression is estimated
without penalty by linear programming (HiGHS solver).

The calibration data are the model's own out-of-sample point forecasts, and the
corresponding realised values, for the 365 days preceding each quarterly re-estimation
date. Point forecasts from October to December 2020 are produced for this purpose and
are not scored, so that the first test quarter has one quarter of calibration data,
rising to a full year from the fourth quarter of 2021. The regressions are
re-estimated at every quarterly re-estimation date. The seven predicted quantiles are
sorted in ascending order, so that no lower quantile forecast exceeds a higher one.

The specification was fixed before any results were computed and was not adjusted
afterwards.

The alternative used in an earlier version of this study, fixed residual offsets, adds
to each point forecast the empirical quantiles of the forecast errors on the 90 days
preceding re-estimation, separately for each hour and spread. The width and shape of
the resulting predictive distribution are therefore the same for every forecast of a
given hour and spread within a quarter. Its results are reported in Section 6.5.2 for
comparison.

Because the post-processing re-estimates the median as well as the interval, it
also changes the mean absolute error of the benchmark models.

---

## 5. Evaluation protocol

### 5.1 Walk-forward design

The protocol comprises 23 quarterly re-estimation dates from January 2021 to July
2026. At each date, every model is estimated on all observations preceding that date
and is used to forecast each day of the following quarter. Concatenating these
quarters yields 1,988 out-of-sample forecast days.

### 5.2 Scoring and aggregation

Pinball loss is the primary criterion. It is the scoring rule for quantile forecasts,
also termed quantile loss or the check function, introduced by Koenker and Bassett
(1978). For a realised value y and a forecast ŷ at level q, the loss is

    L_q(y, ŷ) = q · (y − ŷ)        if y ≥ ŷ
              = (1 − q) · (ŷ − y)   if y < ŷ

The reported pinball loss is the arithmetic mean of L over forecast days, the 24
delivery hours, both spreads and all seven quantile levels. Lower values indicate
better forecasts. The rule is strictly proper for quantiles: expected loss is
minimised by reporting the forecaster's true quantiles.

Mean absolute error (MAE), the arithmetic mean of |y − ŷ| where ŷ is the predicted
0.50 quantile, is reported per spread, in EUR/MWh, alongside pinball loss for
comparability with point-forecast studies.

Tables report the mean over the 23 quarters of each quarter's mean loss. For the deep
learning models this is further averaged over three seeds: 0, 1 and 2. A seed is the
integer that initialises the pseudo-random number generator, which determines a neural
network's initial weights, the order in which training examples are presented and the
dropout mask; two runs that differ only in seed produce different fitted models. The
benchmark models are deterministic. The median over quarters is also reported, because
the mean is sensitive to the six quarters of the 2022 energy crisis.

All statistical tests are applied to daily aggregate losses, one value per forecast
origin, using seed 0 for the deep learning models. Treating 47,712 hourly errors as
independent when 1,988 independent forecasts were issued would overstate the
effective sample size by a factor of 24.

**Diebold–Mariano test** (Diebold and Mariano, 1995). Tests the null hypothesis of
equal expected loss between two forecasts. The long-run variance of the loss
differential is estimated by the heteroskedasticity- and autocorrelation-consistent
estimator of Newey and West (1987).

**Crisis period.** The six quarters from October 2021 to March 2023 are reported
separately, corresponding to the European energy crisis.

### 5.3 The seed-variability threshold

The seed-variability threshold is the median, taken across models, of the
within-quarter standard deviation of mean absolute error across seeds. A difference
between two models smaller than this threshold cannot be distinguished from the
variation produced by re-estimating a single model with a different seed. The quantity
is referred to in the source code and in `RESULTS.md` as the "noise floor". It is
computed over every model configuration estimated in the study, including variants not
reported here, using seeds 0, 1 and 2, which cover every quarter for every
configuration. The threshold is 0.59 on DE–FR and 0.69 on DE–PL. Recomputing it on
DE–FR using only the first k seeds gives 0.37 at k = 2, 0.59 at k = 3, 0.62 at k = 5
and 0.64 at k = 8, so the estimate changes little beyond three seeds.

---

## 6. Results

### 6.1 Principal results

Quarterly re-estimation for all models; benchmarks with quantile regression
post-processing.

| Model | Pinball (mean) | Pinball (median quarter) | DE–FR MAE | DE–PL MAE |
|---|---|---|---|---|
| `lear` | **6.45** | **5.29** | **18.19** | 23.00 |
| `mlp_pool` | 6.80 | 5.36 | 20.02 | **22.21** |
| `naive` | 7.79 | 6.53 | 22.07 | 25.64 |
| `transformer` | 8.21 | 6.01 | 19.08 | 24.83 |
| `lstm` | 8.50 | 5.86 | 19.70 | 25.02 |

Diebold–Mariano tests on daily pinball loss (seed 0, T = 1,988); a negative statistic
indicates that the first model has lower loss:

| Pair | Statistic | p | Lower loss |
|---|---|---|---|
| `lear` vs `mlp_pool` | −1.92 | 0.055 | neither at 5% |
| `lear` vs `transformer` | −7.08 | < 0.0001 | `lear` |
| `lear` vs `lstm` | −7.97 | < 0.0001 | `lear` |
| `lear` vs `naive` | −11.09 | < 0.0001 | `lear` |
| `mlp_pool` vs `transformer` | −6.77 | < 0.0001 | `mlp_pool` |
| `mlp_pool` vs `lstm` | −7.83 | < 0.0001 | `mlp_pool` |
| `mlp_pool` vs `naive` | −7.54 | < 0.0001 | `mlp_pool` |
| `transformer` vs `lstm` | −2.23 | 0.026 | `transformer` |
| `naive` vs `transformer` | −2.59 | 0.010 | `naive` |
| `naive` vs `lstm` | −3.81 | 0.0001 | `naive` |

LEAR has the lowest mean pinball loss, the lowest median-quarter pinball loss and the
lowest DE–FR mean absolute error; the pooled MLP has the lowest DE–PL mean absolute
error. The difference between them in pinball loss is not significant at the 5% level,
and each has significantly lower pinball loss than each of the other three models.
LEAR has a lower quarterly pinball loss than the pooled MLP in 14 of the 23 quarters.

On DE–FR, LEAR's advantage over the transformer (0.89) exceeds the seed-variability
threshold of 0.59. On DE–PL, the pooled MLP's advantage over LEAR (0.79)
exceeds the threshold of 0.69 by a small margin. The model with the lowest error
therefore differs between the two spreads.

### 6.2 Attention and recurrence

| Model | Pinball | Parameters |
|---|---|---|
| `mlp_pool` | 6.80 | 43,936 |
| `transformer` | 8.21 | 79,550 |
| `lstm` | 8.50 | 48,942 |

Among the three deep learning models, which share inputs, training objective,
estimation procedure and output layer, the transformer's pinball loss is 20.7% higher
than that of a multilayer perceptron that encodes no sequential structure, at 1.8
times the parameter count. The recurrent network performs worse still. Both are
significantly worse than the pooled MLP and than LEAR (Section 6.1).

On the distributional criterion, both are also significantly worse than the naive
forecast given quantile regression post-processing. This result does not hold on
point accuracy: on DE–FR mean absolute error the transformer (19.08) is
substantially more accurate than the naive forecast (22.07), and on the median
quarter both sequence models have lower pinball loss than the naive forecast (6.01
and 5.86 against 6.53). The naive forecast's advantage in mean pinball loss arises
in the crisis quarters (Section 6.4).

### 6.3 Calibration

For quantile level q, coverage is the proportion of realised values that fall at or
below the forecast q-quantile. A correctly calibrated forecast has coverage equal to
q; coverage below q at the upper levels indicates that the predictive distribution is
too narrow. The table reports coverage over the 1,988 test days (seed 0 for the deep
learning models); the correct value for each column is the quantile level in the
header.

| Model | 0.05 | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|
| `lear`, post-processed | 0.061 | 0.107 | 0.235 | 0.458 | 0.706 | 0.857 | 0.918 |
| `lear`, fixed offsets | 0.084 | 0.125 | 0.247 | 0.460 | 0.678 | 0.825 | 0.883 |
| `naive`, post-processed | 0.062 | 0.106 | 0.235 | 0.493 | 0.713 | 0.851 | 0.915 |
| `mlp_pool` | 0.090 | 0.112 | 0.223 | 0.417 | 0.630 | 0.780 | 0.843 |
| `transformer` | 0.147 | 0.183 | 0.281 | 0.435 | 0.599 | 0.718 | 0.774 |
| `lstm` | 0.143 | 0.182 | 0.267 | 0.430 | 0.597 | 0.718 | 0.776 |

All models place their 0.95 quantile too low. The post-processed benchmarks are
closest to nominal coverage at both tails. The transformer and recurrent network are
furthest from it: their 0.05 quantile lies above the realised value in 14–15% of
cases rather than 5%, and their 0.95 quantile lies above it in 77–78% of cases rather
than 95%. Their predictive intervals are too narrow in both directions.

### 6.4 Crisis quarters

Pinball loss, mean over quarters:

| Model | Crisis (6 quarters, Oct 2021 – Mar 2023) | Other (17 quarters) |
|---|---|---|
| `lear` | 11.59 | **4.64** |
| `mlp_pool` | **11.55** | 5.12 |
| `naive` | 13.96 | 5.61 |
| `transformer` | 16.51 | 5.28 |
| `lstm` | 16.99 | 5.51 |

LEAR and the pooled MLP are level in the crisis quarters; LEAR has the lower
loss outside them. The sequence models' deficit is concentrated in the crisis
quarters.

### 6.5 Checks on the benchmark

The principal comparison is made against LEAR, so its conclusions depend on LEAR
having been set up fairly. LEAR differs from the deep learning models in two ways that
could affect the comparison. First, it produces a single predicted value for each hour
and needs a separate post-processing step to produce quantile forecasts; the deep
learning models produce quantile forecasts directly. Second, in the principal results
it is re-estimated once per quarter, whereas the published version is re-estimated
every day. The three checks in this section examine these differences. Section 6.5.1
asks whether post-processing gives the benchmarks an advantage that the deep learning
models do not receive. Section 6.5.2 shows how much LEAR's result depends on the
post-processing method. Section 6.5.3 shows the effect of re-estimating LEAR every
day.

#### 6.5.1 Identical post-processing for all models

The principal comparison applies post-processing to the benchmark models only. If
the same post-processing also improved the deep learning models, the comparison would favour
the benchmarks. To test this, every model's central forecast was post-processed
identically — the predicted median for the deep learning models, the point forecast for
LEAR — with calibration data beginning in January 2021, the first date for which
forecasts from the deep learning models exist, and scoring over the 22 quarters from April 2021 (seed 0).

| Model | As trained | With identical post-processing |
|---|---|---|
| `lear` | — | 6.65 |
| `mlp_pool` | 6.94 | 6.96 |
| `transformer` | 8.67 | 8.51 |
| `lstm` | 9.00 | 8.64 |

Post-processing does not improve the pooled MLP and improves the
transformer and recurrent network only slightly. Over this window LEAR has lower
pinball loss than the pooled MLP in its as-trained form (Diebold–Mariano
−2.17, p = 0.030, T = 1,901). The principal conclusion does not depend on applying
post-processing to the benchmarks alone.

#### 6.5.2 Benchmarks with fixed residual offsets

LEAR produces a single predicted value for each hour, but every model in this study is
scored on seven quantile forecasts. LEAR's single value must therefore be turned into
seven quantile forecasts by a separate step, and there is more than one way to do
this. The principal results use quantile regression post-processing. This section
repeats the scoring with a simpler method, fixed residual offsets (Section 4.4), in
order to show how much LEAR's score, and its ranking against the deep learning models,
depends on which method is used. The table gives the pinball loss of the two benchmark
models under each method.

| Model | Fixed residual offsets | Quantile regression post-processing |
|---|---|---|
| `lear` | 8.40 | **6.45** |
| `naive` | 9.62 | **7.79** |

The point forecasts are the same under both methods; only the predictive distribution
differs. With fixed residual offsets, LEAR's pinball loss (8.40) is higher than that
of the pooled MLP (6.80) and the transformer (8.21), so that the deep learning models
would appear to outperform the benchmark. The comparison between the benchmark and the
deep learning models therefore depends on how the benchmark's predictive distribution
is constructed.

#### 6.5.3 Daily re-estimation of LEAR

In the principal results every model, including LEAR, is re-estimated once per
quarter, so that all models are treated in the same way. In the benchmark published by
Lago et al. (2021), LEAR is instead re-estimated every day, on four calibration
windows whose forecasts are averaged (Section 2.1). The quarterly schedule may
therefore understate how accurate LEAR can be. This section measures the size of that
effect by also re-estimating LEAR every day. Only the longest of the four published
windows, the most recent four years of data, is used. The post-processing is unchanged
and is still re-estimated quarterly.

| LEAR re-estimation | Pinball loss |
|---|---|
| Quarterly | 6.45 |
| Daily | 5.90 |

Daily re-estimation reduces pinball loss from 6.45 to 5.90 (Diebold–Mariano −9.34,
p < 0.0001). The deep learning models were not re-estimated daily, because one fit takes
37 to 107 seconds against approximately 6 seconds for LEAR, which amounts to 20 to 60
hours per seed over the test period. This result therefore does not compare
architectures. It shows that re-estimation frequency has a large effect on LEAR, and
that LEAR's low estimation cost permits a re-estimation frequency that the deep
learning models in this configuration do not.

---

## 7. Discussion

### 7.1 Attention and recurrence

On these spreads, neither the transformer nor the LSTM network improves on the pooled
MLP or on LEAR. The transformer is second-best on DE–FR mean absolute error, so part
of its deficit lies in its quantile forecasts, which are too narrow (Section 6.3).

This agrees with Lago et al. (2018), where an MLP with two hidden layers outperformed
recurrent networks. It differs from the earlier studies in two respects (Section 2.3):
there, recurrent networks outperformed a LEAR-type model (Lago et al., 2018), and
transformers matched or outperformed LEAR-type and MLP models (Wang et al., 2024;
Elashhab et al., 2026). The result is therefore potentially new, but it is not
explained. Five untested explanations are possible:

1. **Tuning and size.** The deep learning models were not tuned, and the transformer
   is smaller than published configurations (model dimension 48, two encoder layers).
   The models in Lago et al. (2018) and Elashhab et al. (2026) were tuned, and in the
   latter study tuning removed most of the transformer's deficit.
2. **Input representation.** The transformer represents each history day by one token
   and cannot distinguish the hours within it (Section 4.3).
3. **Test period.** The deficit is concentrated in the crisis quarters of 2021–2023
   (Section 6.4), whereas Lago et al. (2018) and Elashhab et al. (2026) were tested on
   calmer periods, and Lago et al. (2018) and Wang et al. (2024) re-estimated their
   models daily rather than quarterly.
4. **Quantity forecast.** Spreads, unlike price levels, are often exactly zero and
   occasionally very large.
5. **Evaluation criterion.** The earlier studies score point forecasts only; pinball
   loss also scores the width of the forecast distribution.

Further experiments are needed to distinguish these explanations: a hyperparameter
search that includes larger transformers and hourly tokens; separate evaluation
before, during and after the crisis, with more frequent re-estimation; and the same
comparison on price levels. Until then, the result applies to the configurations
tested here, not to transformers or recurrent networks in general.

### 7.2 Narrow quantile forecasts

The deep learning models' quantile forecasts are too narrow (Section 6.3), even though
the models are trained on pinball loss. One possible reason, not tested, is that their
ranges vary only in ways learned from past data, whereas the benchmarks'
post-processing widens the range in proportion to the size of the point forecast,
including for forecasts larger than any seen before. The difference would matter most
in the crisis quarters. The narrow ranges do not account for the ranking of the
models: with the same post-processing as the benchmarks, the deep learning models'
pinball loss changes little and LEAR remains ahead (Section 6.5.1).

---

## 8. Conclusion

For day-ahead DE–FR and DE–PL price spreads, using only information expected to be
public before the auction closed and with every model re-estimated quarterly, the
standard linear benchmark LEAR and a multilayer perceptron with pooled inputs are
jointly the most accurate models, with pinball losses of 6.45 and 6.80, which are
statistically indistinguishable from each other. A transformer and a recurrent
network, matched to the pooled MLP in inputs, training objective and approximate size,
are significantly less accurate than both, and on the distributional criterion are
also less accurate than a naive forecast with post-processing. Their quantile
forecasts are too narrow at both tails.

This result differs from most earlier comparisons on day-ahead price levels (Lago et
al., 2018; Wang et al., 2024; Elashhab et al., 2026). It was obtained with deep
learning models that were not tuned, and further experiments are required to determine
whether it reflects the architectures, their configuration, the test period or the
quantity forecast (Section 7.1).

---

## References

Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of
Business & Economic Statistics* 13(3), 253–263.

Elashhab, H., Papineni, S. S., Dorn, M., Hagenmeyer, V. and Schäfer, B. (2026). Deep
learning for cross-border electricity price forecasting: a comparative study.
arXiv:2608.17091. https://arxiv.org/abs/2608.17091

Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica* 46(1),
33–50.

Lago, J., De Ridder, F. and De Schutter, B. (2018). Forecasting spot electricity
prices: deep learning approaches and empirical comparison of traditional algorithms.
*Applied Energy* 221, 386–405. https://doi.org/10.1016/j.apenergy.2018.02.069

Lago, J., Marcjasz, G., De Schutter, B. and Weron, R. (2021). Forecasting day-ahead
electricity prices: a review of state-of-the-art algorithms, best practices and an
open-access benchmark. *Applied Energy* 293.
https://www.sciencedirect.com/science/article/pii/S0306261921004529 ·
https://github.com/jeslago/epftoolbox

Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*
55(3), 703–708.

Nowotarski, J. and Weron, R. (2015). Computing electricity spot price prediction
intervals using quantile regression and forecast averaging. *Computational Statistics* 30(3), 791–803.

Wang, L., Liu, J., Zhang, H. and Wang, L. (2024). Revisiting day-ahead electricity
price: simple model save millions. arXiv:2405.14893. https://arxiv.org/abs/2405.14893

---

## Appendix A: Reproduction

```
python data/build_dataset.py --source <modelling_table.csv>   # construct the dataset
python -m pytest tests/                                        # leakage and model tests
python harness/walkforward.py --models mlp_pool lstm transformer --seeds 0 1 2
python experiments/benchmarks.py points                        # quarterly LEAR and naive
python experiments/benchmarks.py daily --worker 0 --nworkers 2 # daily LEAR (Section 6.5.3)
python experiments/benchmarks.py daily --worker 1 --nworkers 2
python experiments/benchmarks.py merge
python experiments/benchmarks.py post                          # post-processing
python experiments/benchmarks.py common                        # Section 6.5.1
python experiments/paper_tables.py                             # every table in the paper
```

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` when running several processes
in parallel; without this, LEAR estimation slowed by more than a factor of ten in
testing because of thread contention.

## Appendix B: Companion documents

`LITERATURE.md` — the literature review.

`RESULTS.md` — detailed results from the original configuration. Its benchmark comparisons use fixed residual offsets and
are superseded by Section 6 of this paper.

