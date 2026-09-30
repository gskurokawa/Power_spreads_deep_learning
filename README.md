# Deep learning for European power price spreads

Code for the study in `PAPER.md`: day-ahead forecasts of the DE–FR and DE–PL
price spreads, as seven quantiles per hour, comparing a naive forecast, LEAR,
a pooled MLP, an LSTM network and a transformer.

## Layout

| Folder | Contents |
|---|---|
| `data/` | `feature_classes.py`: which inputs may be used, and up to when (Section 3.2). `build_dataset.py`: builds `dataset.npz` from the source table. |
| `models/` | `baselines.py`: naive forecast and LEAR. `mlp.py`, `lstm.py`, `transformer.py`: the deep learning models. `nn_common.py`: parts they share. |
| `harness/` | `walkforward.py`: quarterly re-estimation and scoring. `train.py`: training loop for the deep learning models. `dataset.py`: data loader and windows. `metrics.py`: pinball loss, MAE. `dmtest.py`: Diebold–Mariano test. |
| `experiments/` | `benchmarks.py`: benchmark point forecasts, daily LEAR, and quantile regression post-processing. `paper_tables.py`: every table in the paper. Results files: `results.csv`, `daily/`, `daily_fixed/`, `daily_common/`, `points/`. |
| `tests/` | Leakage test and model tests. |

## Reproduction

```
python data/build_dataset.py --source <modelling_table.csv>
python -m pytest tests/
python harness/walkforward.py --models mlp_pool lstm transformer --seeds 0 1 2
python experiments/benchmarks.py points
python experiments/benchmarks.py daily --worker 0 --nworkers 2
python experiments/benchmarks.py daily --worker 1 --nworkers 2
python experiments/benchmarks.py merge
python experiments/benchmarks.py post
python experiments/benchmarks.py common
python experiments/paper_tables.py
```

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` when running several
processes in parallel; otherwise LEAR estimation slows by more than a factor of
ten because of thread contention.

The deep learning model files still contain the linear-skip and flattened-history
variants that were explored earlier (their `CONFIGS` entries and the `skip` and
`history` options). The paper reports only `mlp_pool`, `lstm` and `transformer`.
