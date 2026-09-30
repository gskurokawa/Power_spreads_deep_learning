"""Smoke tests for the torch models. RUN THIS BEFORE THE PHASE 2 GRID.

The environment this code was written in has no torch, so none of the neural
code has ever executed. These checks are the substitute for that: they verify
shapes, that the loss agrees with the numpy implementation the results are scored
with, that quantiles cannot cross, that the model can overfit a small batch (so
gradients actually flow), and that seeding is deterministic. Then it times one
fit, which is the number the Phase 6 compute budget depends on.

    python tests/test_torch_models.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "models"))

import torch                                                   # noqa: E402
from dataset import SpreadData                                 # noqa: E402
import metrics                                                 # noqa: E402
import mlp                                                     # noqa: E402
from nn_common import PinballLoss, count_params, set_seed      # noqa: E402
from train import Standardiser, TorchModel                     # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILS.append(name)


def main():
    data = SpreadData(os.path.join(ROOT, "data"))
    idx = np.arange(64)
    raw = [getattr(data, b)[idx] for b in
           ("X_auction", "X_realised", "X_ahead", "X_calendar")]

    # Standardise, exactly as TorchModel.fit does. The models assume standardised
    # inputs -- LinearSkip's initialisation in particular is scaled for unit
    # variance. Feeding raw values (load is ~50,000 MW) makes the untrained
    # network emit numbers in the thousands, which is a property of the input
    # scale, not a bug in the model. An earlier version of this file skipped the
    # standardiser and failed checks 3 and 5 for exactly that reason.
    blocks = [torch.tensor(a, dtype=torch.float32)
              for a in Standardiser().fit(raw).transform(raw)]
    y = torch.tensor(data.Y[idx], dtype=torch.float32)

    print("\n1. parameter budgets")
    for name, cfg in mlp.CONFIGS.items():
        n = count_params(mlp.build(data, **cfg))
        print(f"     {name:18s} {n:>9,d}")
    check("pooled MLP stays under 150k params",
          count_params(mlp.build(data, **mlp.CONFIGS["mlp_pool"])) < 150_000)
    # The skip must not dominate: a dense raw->output map would be 2.04M params,
    # which turns every "with skip" run into a linear model wearing a small net.
    bare = count_params(mlp.build(data, **mlp.CONFIGS["mlp_pool"]))
    withskip = count_params(mlp.build(data, **mlp.CONFIGS["mlp_pool_skip"]))
    check("linear skip adds less than half the base model's parameters",
          withskip - bare < 0.5 * bare, f"skip adds {withskip - bare:,d} vs base {bare:,d}")

    print("\n2. shapes")
    set_seed(0)
    m = mlp.build(data, **mlp.CONFIGS["mlp_pool_skip"]).eval()
    out = m(*blocks)
    check("output is (B, 24, 2, 7)", tuple(out.shape) == (64, 24, 2, 7), str(tuple(out.shape)))

    print("\n3. loss agrees with the numpy scorer")
    tl = float(PinballLoss()(out, y).detach())
    nl = metrics.pinball(y.numpy(), out.detach().numpy())
    # RELATIVE tolerance: these are float32 reductions over 64*24*2*7 elements,
    # so agreement to ~1e-7 relative is exact for practical purposes. An absolute
    # 1e-5 bound is meaningless once the loss is O(1000).
    rel = abs(tl - nl) / max(abs(nl), 1e-9)
    check("torch pinball == numpy pinball", rel < 1e-5, f"{tl:.8f} vs {nl:.8f} (rel {rel:.2e})")

    print("\n4. quantiles cannot cross")
    d = np.diff(out.detach().numpy(), axis=-1)
    check("quantiles monotone non-decreasing", (d >= -1e-6).all(), f"min diff {d.min():.2e}")

    print("\n5. the linear skip starts as a near no-op")
    # .eval() is essential, and its absence was a real bug in an earlier version
    # of this file. A module built by mlp.build() is in TRAIN mode, so nn.Dropout
    # is live and draws fresh masks at forward time. Constructing the skip
    # consumes RNG draws, so the with-skip model reached its forward pass with a
    # different RNG state and therefore different dropout masks. The 0.62 "skip
    # delta" that produced was dropout noise -- 200x the skip's actual
    # contribution, which arithmetic puts at ~0.003 before the output transform
    # and ~0.02 after it. In eval() the comparison isolates the skip, as intended.
    set_seed(0); a = mlp.build(data, history="pool", skip=False).eval()(*blocks)
    set_seed(0); b = mlp.build(data, history="pool", skip=True).eval()(*blocks)
    delta = float((a - b).abs().max().detach())
    check("skip barely moves the untrained output", delta < 0.1, f"max delta {delta:.4f}")

    print("\n6. gradients flow -- can it overfit 32 samples?")
    set_seed(0)
    m = mlp.build(data, history="pool", skip=False, dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lossf = PinballLoss()
    small = [b[:32] for b in blocks]
    ys = y[:32]
    first = float(lossf(m(*small), ys).detach())
    for _ in range(300):
        opt.zero_grad(); l = lossf(m(*small), ys); l.backward(); opt.step()
    last = float(l.detach())
    check("loss falls by >70% when overfitting a tiny batch",
          last < 0.3 * first, f"{first:.4f} -> {last:.4f}")

    print("\n7. determinism")
    set_seed(7); o1 = mlp.build(data, **mlp.CONFIGS["mlp_pool"]).eval()(*blocks)
    set_seed(7); o2 = mlp.build(data, **mlp.CONFIGS["mlp_pool"]).eval()(*blocks)
    check("same seed gives identical weights", torch.allclose(o1, o2))
    set_seed(8); o3 = mlp.build(data, **mlp.CONFIGS["mlp_pool"]).eval()(*blocks)
    check("different seed gives different weights", not torch.allclose(o1, o3))
    # And that dropout IS live in train mode -- the property that broke check 5.
    set_seed(9); mt = mlp.build(data, **mlp.CONFIGS["mlp_pool"]).train()
    check("dropout is active in train mode",
          not torch.allclose(mt(*blocks), mt(*blocks)))

    print("\n8. the two history streams are kept separate")
    m = mlp.build(data, **mlp.CONFIGS["mlp_pool"])
    check("auction encoder sized for auction features",
          m.enc_auction.lin.in_features == len(data.cols["auction"]))
    check("realised encoder sized for realised features",
          m.enc_realised.lin.in_features == len(data.cols["realised"]))

    print("\n9. end-to-end fit/predict, and the timing number Phase 6 needs")
    tr, va, te = data.windows("2025-01-01", policy="rolling", years=2)
    print(f"     train {len(tr)}  val {len(va)}  test {len(te)}")
    for cfg_name in ("mlp_pool_skip", "mlp_flat_skip"):
        cfg = mlp.CONFIGS[cfg_name]
        t0 = time.time()
        tm = TorchModel(lambda c=cfg: mlp.build(data, **c), cfg_name, seed=0).fit(data, tr, va)
        secs = time.time() - t0
        pred = tm.predict(data, te)
        s = metrics.summary(data.Y[te], pred)
        print(f"     {cfg_name:16s} {secs:6.1f}s  {tm.epochs_run:3d} epochs  "
              f"{tm.n_params:>8,d} params  pinball {s['pinball']:.3f}  "
              f"DE-FR MAE {s['spread_DE_FR_mae']:.2f}")
        check(f"{cfg_name} predicts the right shape",
              pred.shape == (len(te), 24, 2, 7), str(pred.shape))
        print(f"     -> 46 fits x 8 seeds = {46*8*secs/3600:.1f} h for this model alone")

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED -- safe to run the Phase 2 grid")


if __name__ == "__main__":
    main()
