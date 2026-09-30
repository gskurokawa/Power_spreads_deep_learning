# Literature review — deep learning for European power price spreads

Searched September 2026. Papers I actually opened and read are marked ✅; ones I only saw in
search listings are marked ○ and should be read before citing.

---

## Bottom line

**Nobody appears to have done exactly this.** Electricity price forecasting (EPF) is a large,
mature literature with a standard benchmark. Cross-border *information* has been used to forecast
price *levels*. Price *spreads* have been modelled carefully — but in US day-ahead/real-time
markets, not European cross-zonal ones, and with distributional rather than deep-learning methods.
The intersection you're proposing — European cross-zonal spreads, architecture comparison,
gate-closure-audited — is open.

That's the good news. The bad news is that three of your plan's assumptions need revising, and
one of my earlier claims to you was wrong.

---

## 1. The benchmark you should be using

✅ **Lago, Marcjasz, De Schutter & Weron (2021), "Forecasting day-ahead electricity prices: a
review of state-of-the-art algorithms, best practices and an open-access benchmark",
*Applied Energy*.** Ships as [`epftoolbox`](https://github.com/jeslago/epftoolbox).

This is the reference point for the entire field. It defines five open datasets (Nord Pool, PJM,
EPEX-BE, EPEX-FR, EPEX-DE), a standard two-year test period, and two benchmark models:

- **LEAR** — Lasso Estimated AutoRegressive, the linear benchmark
- **DNN ensemble** — the neural benchmark

**Plan change:** replace the plain linear layer in my rung 2 with **LEAR**. It's a far stronger
linear baseline than an unregularised flattened-window regression, it's the model every EPF reader
knows, and it's already implemented in `epftoolbox`. Keeping a naive linear layer alongside it is
fine but LEAR is the one that carries argumentative weight. It also makes your results legible to
anyone in the field — "we beat LEAR" means something; "we beat a linear layer I wrote" doesn't.

Also adopt the field's significance test: **Diebold-Mariano** on forecast error differences. My
plan proposed seed distributions, which is necessary but not what EPF reviewers expect. Do both.

---

## 2. On transformers — I was too pessimistic

✅ **"A Transformer approach for Electricity Price Forecasting" (arXiv:2403.16108).**

A pure transformer, no recurrence, evaluated on all five `epftoolbox` datasets against Lago's DNN
ensemble and a naive model. Reported MAE: Nord Pool 2.33 vs DNN 3.40; PJM 3.67 vs 7.17; EPEX-BE
6.54 vs 9.38. Significantly better by Diebold-Mariano (p < 0.05) on four of five datasets.

I told you my prior was that the transformer would lose. On this evidence that prior was too
strong for EPF specifically, and you should know that going in. Two caveats keep it from settling
the question: those are price *levels*, not spreads, and a single paper reporting large wins over
a benchmark is exactly the situation where independent replication matters — which is, usefully,
part of what your project would be.

○ **Yu et al. (2026), "Deep Learning for Electricity Price Forecasting: A Review of Day-Ahead,
Intraday, and Balancing Electricity Markets" (arXiv:2602.10071).** I read this one; it's thinner
than its title suggests. It documents attention models appearing from 2021 onward but offers no
head-to-head benchmarking, and notes that many surveyed studies don't even specify their loss
function, "complicating direct comparison across models." Useful mainly as evidence that **the
comparison you want to run has not been run systematically.** That's a citable gap.

---

## 3. But linear models remain competitive

✅ **El Mahtouta & Ziel (2026), "Electricity Price Forecasting: Bridging Linear Models, Neural
Networks and Online Learning" (arXiv:2601.02856).** German-Luxembourg and Spanish markets,
2018-2025 hourly.

Ziel is one of the significant names in this field, so this carries weight. Their finding:
"linear models generally exhibit competitive performance with minimal computational requirements"
while nonlinear models "improve forecasting accuracy with a surge in computational costs." Their
proposed model isn't a pure neural network — it's a **hybrid with linear skip connections** feeding
alongside nonlinear hidden layers, so the network only has to learn the residual nonlinearity.
Reported 11-12% RMSE and 14-17% MAE reductions over benchmarks.

**Plan change:** add a linear skip connection to the MLP and transformer. It costs one line, it's
an established result rather than an invention of yours, and on ~1,800 training days it's exactly
the kind of inductive-bias assist that should help. It also gives you a clean ablation: does the
attention block add anything *once the linear part is already handled*? That's a sharper version
of your research question than the one in the plan.

---

## 4. Cross-border information — done, for levels

✅ **Mascarenhas, De Blauwe, Amelin & Kazmi (2026), "Leveraging asynchronous cross-border market
data for improved day-ahead electricity price forecasting in European markets", *Applied Energy*
404.**

The closest published work to your setup. Exploits the fact that bidding zones have **different
gate closure times** — German, Luxembourg, Austrian and Swiss prices publish earlier, so they're
legitimately available when forecasting Belgium and Sweden SE3. Reports 22% accuracy improvement
for Belgium, 9% for Sweden; MAE 10.18 and 11.79 €/MWh. Holds under both normal and extreme price
conditions. Also finds that more market data doesn't monotonically help.

Critically for you: **it forecasts price levels, not spreads.** The cross-border angle is taken;
the spread target is not.

The asynchronous-gate-closure idea is worth stealing outright. If any zone in your feature set
closes before DE/FR/PL, its published price is a legitimate pre-auction feature and is probably a
strong one. Worth an explicit check of gate closure times across your zones — it may hand you a
feature nobody in your comparison has.

○ Per Yu et al.'s review, graph neural networks have been applied to "model spatial dependencies
across interconnected bidding zones" (attributed to Meng et al., 2024, multi-view fusion
spatio-temporal GNN). I haven't read these directly. If your multi-zone transformer arm is the
interesting one, a GNN is its natural rival and a reviewer will ask. Worth reading before you
commit to the multi-zone framing.

---

## 5. Spread forecasting — the real analogue is American

European cross-zonal spreads barely appear. The developed literature is on **DART spreads** (US
day-ahead minus real-time), which is structurally the same problem: a difference that sits near
zero most of the time with violent tails.

✅ **Forgetta, Godin & Augustyniak (2025), "Distributional forecasting of electricity DART spreads
with a covariate-dependent mixture model", *Energy Economics* 144.** NYISO Long Island, hourly.

They model the spread as a **three-regime mixture** — Gaussian for normal conditions, plus two
Generalised Pareto components for positive and negative spikes — because no single distribution
fits. Covariates: load forecasts, weather, natural gas futures. Benchmarked against **deep neural
network quantile regression**, and the mixture model *won* on out-of-sample quantile prediction.
Gas futures were the strongest predictor of spread volatility.

This is the most directly transferable paper in the review, and it changes a plan decision. I had
the zero-inflation hurdle model as a Phase 5 secondary experiment. Given that a well-specified
mixture beat a DNN on a structurally identical target, **the distributional treatment should be a
first-class arm, not an afterthought** — and your models should probably emit distributions
(quantiles) rather than point forecasts. Your target is 30% exact zeros; a point forecast under
MAE will simply learn to hug zero.

○ Galarneau-Vincent, Gauthier & Godin, "Foreseeing the worst: forecasting electricity DART
spikes", *Energy Economics* (2023) — same group, spike-prediction framing.
○ "Trading Electrons: Predicting DART Spread Spikes in ISO Electricity Markets" (arXiv:2601.05085)
— recent, unread.

The gap to claim: **European market-coupled zonal spreads have a structural feature DART spreads
don't** — the exact-zero point mass created by market coupling, which is a *deterministic*
consequence of an uncongested interconnector rather than a statistical coincidence. Nobody seems
to have exploited that. It's a genuine research angle: the zero isn't noise, it's information
about congestion, and a model that knows the coupling mechanism should beat one that doesn't.

---

## 6. Leakage — there's a name for the discipline you need

✅ **Rokicki, Bórawski, Bełdycka-Bórawska & Klepacki (2026), "Predicting Negative Day-Ahead
Electricity Prices Across 12 European Bidding Zones: A Gate-Closure-Audited Explainable Machine
Learning Framework", *Applied Sciences* 16(18), 9063.**

Twelve zones including DE_LU, FR, and your neighbours, 2019-2025, 736,416 observations. Their
**"gate-closure audit"** is precisely the discipline my plan called for, with a citable name and a
clean formulation: classify every feature by its official publication deadline, then split into a
**main pre-auction model** (load forecasts, transfer capacity, calendar, completed-auction price
history) and a **diagnostic layer** (wind/solar forecasts, post-clearing scheduled exchanges) used
only for retrospective analysis, never for operational forecasting.

Adopt the terminology and the two-layer structure verbatim. It maps exactly onto your existing
`--pre-auction` flag, and it means your methodology section cites an established practice rather
than asserting your own care.

Two further transferable details. Their strongest pre-auction SHAP features were the previous
completed day's minimum price (1.316), weekend flag (0.580), and the 24-hour price lag (0.511) —
worth ensuring all three are in your feature set. And they flag a limitation you share: the ENTSO-E
API doesn't let you reconstruct historical forecast *vintages*, so publication-rule auditing
substitutes for verified per-observation timestamps. State that caveat explicitly; it's honest and
it's the same one a published paper had to make.

Note also that on their rare-event target, XGBoost (PR-AUC 0.380) crushed penalised logistic
regression (0.081). For the congested/uncongested classification half of a hurdle model, that's a
relevant prior — the nonlinearity may matter more in the classification step than in the
regression step.

---

## 7. Amendments to PLAN.md

| # | Change | Source |
|---|---|---|
| 1 | Replace plain linear rung with **LEAR** from `epftoolbox`; keep naive linear as a sanity check | Lago et al. 2021 |
| 2 | Add **Diebold-Mariano tests** alongside seed distributions | field standard |
| 3 | Add **linear skip connections** to MLP and transformer; ablate | El Mahtouta & Ziel 2026 |
| 4 | Promote the **hurdle/mixture model to a primary arm**; emit quantiles, not point forecasts | Forgetta et al. 2025 |
| 5 | Adopt **"gate-closure audit"** terminology and the main/diagnostic two-layer split | Rokicki et al. 2026 |
| 6 | Check **gate closure times** across zones; earlier-closing zones' prices are legal features | Mascarenhas et al. 2026 |
| 7 | Add prev-day **minimum price** and confirm 24h lag and weekend flag are present | Rokicki et al. 2026 |
| 8 | Read the **GNN** multi-zone work before committing to the multi-zone transformer framing | Yu et al. 2026 |
| 9 | State the **forecast-vintage caveat** explicitly | Rokicki et al. 2026 |

Items 1, 4 and 5 are substantive. The rest are cheap.

---

## 8. The niche, stated precisely

> Deep learning architectures have been compared for European day-ahead price *levels* against a
> standard benchmark, and cross-border information has been shown to improve those forecasts. Price
> *spreads* have been modelled distributionally in US DART markets, where a mixture model beats
> neural quantile regression. But European cross-zonal spreads have a structure neither literature
> addresses: a point mass at exactly zero, generated deterministically by market coupling, whose
> frequency has fallen from 49% to 6% of hours between 2021 and 2026. This project asks whether
> attention-based models offer anything over linear and feedforward alternatives on that target,
> under a gate-closure audit, and whether modelling the coupling mechanism explicitly beats
> treating the spread as a continuous variable.

That is a defensible research question rather than a bake-off, and the regime-shift statistic is
one you measured yourself from data you already hold.

---

## Sources

- [Lago et al. 2021, *Applied Energy* — review + open-access benchmark](https://www.researchgate.net/publication/351198633_Forecasting_day-ahead_electricity_prices_A_review_of_state-of-the-art_algorithms_best_practices_and_an_open-access_benchmark) · [epftoolbox](https://github.com/jeslago/epftoolbox) · [PDF](https://www.bartdeschutter.org/publications/21-011-forecasting-day-ahead-electricity/21-011-forecasting-day-ahead-electricity.pdf)
- [A Transformer approach for Electricity Price Forecasting (arXiv:2403.16108)](https://arxiv.org/html/2403.16108v1)
- [El Mahtouta & Ziel 2026 — Bridging Linear Models, Neural Networks and Online Learning (arXiv:2601.02856)](https://arxiv.org/html/2601.02856v3)
- [Mascarenhas et al. 2026, *Applied Energy* 404 — asynchronous cross-border data](https://www.sciencedirect.com/science/article/abs/pii/S0306261925018070)
- [Rokicki et al. 2026, *Applied Sciences* 16(18) 9063 — gate-closure-audited negative price prediction](https://www.mdpi.com/2076-3417/16/18/9063)
- [Forgetta, Godin & Augustyniak 2025, *Energy Economics* 144 — distributional DART spread forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0140988325001562)
- [Galarneau-Vincent, Gauthier & Godin — Foreseeing the worst: forecasting electricity DART spikes](https://www.sciencedirect.com/science/article/abs/pii/S0140988323000191)
- [Trading Electrons: Predicting DART Spread Spikes in ISO Electricity Markets (arXiv:2601.05085)](https://arxiv.org/html/2601.05085v1)
- [Yu et al. 2026 — Deep Learning for EPF: A Review (arXiv:2602.10071)](https://arxiv.org/html/2602.10071v1)
- [Deep learning-based electricity price forecasting: findings on price predictability in European markets](https://www.sciencedirect.com/science/article/pii/S0360544224026513)

---

# Supplementary search — September 2026 (second pass)

## Why this pass was run

The first pass concluded that "nobody appears to have done exactly this." That
conclusion was drawn from searches on the vocabulary of one subfield: *spread
forecasting*, *cross-border electricity price forecasting*, *DART spread*. It did
**not** search the vocabulary that the adjacent literature actually uses:

- price convergence
- market integration
- price divergence
- congestion forecasting / congestion prediction
- zero-inflated, hurdle
- financial transmission rights, locational marginal price difference

Searching one subfield's terminology and concluding a phenomenon is unstudied is a
predictable failure mode. This pass corrects it.

## The decisive finding

✅ **Ehrenmann? — "Integration in the European electricity market: A machine
learning-based convergence analysis for the Central Western Europe region",
*Energy Policy*** (S0301421519303751). *(Author list not captured from the
abstract page; confirm before citing.)*

Fetched and read in abstract. This paper **builds predictive models of price
convergence versus congestion** in Central Western Europe — the region containing
DE and FR — following the 2015 introduction of flow-based market coupling. Target:
"whether price equalization is reached" between connected markets. Method: random
forests, tested across **sliding-window and aggregate-window training schemes**.
Data 2016–2017.

This is, in substance, the classification component of our hurdle model, published
in 2019, on our region, under the same two window policies we describe as a design
choice in §5.1 of the paper. It is not a minor adjacent reference.

## Other work the first pass missed

○ **Electricity price convergence and dynamics in Europe: market integration,
decarbonisation, electricity mix, and external shocks**, *Energy Efficiency* (2026),
s12053-026-10464-z. Descriptive; the convergence share is its central statistic.

○ **Introducing MEGO and PDC: Novel Indicators for Quantifying Market Rigidity and
Cross-Border Price Divergence in Central European Electricity Markets**, *Applied
Sciences* 16(16), 8343 (2026). Purpose-built indices for cross-border price
divergence.

○ **Forecasting cross-border power transmission capacities in Central Western
Europe using artificial neural networks**, *Energy Informatics* (2019),
s42162-019-0094-y. Forecasts the capacity side of the same mechanism.

○ **Electricity price spike clustering: A zero-inflated GARX approach**, *Energy
Economics* 124 (2023). Zero-inflation applied to spikes rather than spreads;
establishes that the technique is present in the field.

○ **Forecasting the Intra-Day Spread Densities of Electricity Prices**, *Energies*
13(3), 687 (2020). Distributional forecasting of an electricity price spread — a
different spread (intraday vs day-ahead), but the same distributional treatment.

○ **Deep Learning-Based Electricity Price Forecast for Virtual Bidding in Wholesale
Electricity Market**, arXiv:2412.00062. Virtual bidding is DART-spread trading;
worth reading for the loss-function and decision-relevance framing.

○ **Multivariate probabilistic forecasting of electricity prices with trading
applications**, *Energy Economics* (S0140988324007163). Multivariate probabilistic
EPF with an explicit trading objective.

## Revised conclusion on novelty

**The first pass's novelty claim does not survive.** Taking the components of this
project one at a time:

| Component | Prior work |
|---|---|
| The zero mass as a studied phenomenon | Extensive — the convergence literature |
| Predicting convergence vs congestion in CWE | Energy Policy, 2019, random forests |
| Expanding vs rolling window comparison | Same paper |
| Cross-border features for day-ahead EPF | Mascarenhas et al. 2026 |
| Gate-closure audit | Rokicki et al. 2026 |
| Transformer for EPF | arXiv:2403.16108 |
| Linear skip hybrid | El Mahtouta & Ziel 2026 |
| Distributional spread forecasting | Forgetta et al. 2025 (DART); Energies 2020 (intraday) |
| Zero-inflation in electricity price models | Energy Economics 2023 |

Every component has prior work, including the one previously treated as the
distinctive angle — the zero as a congestion signal — which was modelled
predictively in 2019.

What these two passes did **not** find is a study forecasting the *magnitude* of a
European cross-zonal spread as a full predictive distribution, day-ahead, with an
architecture comparison. That is a statement about what two searches found, not a
statement about what exists, and it should not be converted into a novelty claim.

**Recommendation for PAPER.md:** position the study as a replication and comparison
exercise rather than as filling a gap. Its defensible value is (a) an independent
test of arXiv:2403.16108's strong transformer result, on a different target, with
the opposite outcome; (b) the controlled comparison that Yu et al. (2026) state has
not been conducted systematically — their observation, not ours; and (c) the
target-dependence result, which stands on its own evidence regardless of what else
has been published.

## Remaining search gaps

Not yet searched: non-English literature; the ACER and ENTSO-E market monitoring
reports, which publish convergence statistics and may contain forecasting work;
conference proceedings (IEEE PES, EEM); and the FTR/CRR forecasting literature in
depth, which is the US analogue to forecasting congestion rent and is likely to
contain directly comparable distributional work.
