"""Phase 4 checks for the LSTM, plus the last timing number before the grid.

Shorter than the transformer's suite because there is no hand-written mechanism
to verify against a reference -- nn.LSTM is the reference. What matters here is
that it shares the others' output plumbing exactly, so that any difference in
results is attributable to recurrence rather than to how the answer is emitted.

    python tests/test_lstm.py
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
import lstm as L                                               # noqa: E402
import mlp, transformer as T                                   # noqa: E402
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
    blocks = [torch.tensor(a, dtype=torch.float32)
              for a in Standardiser().fit(raw).transform(raw)]
    y = torch.tensor(data.Y[idx], dtype=torch.float32)

    print("\n1. parameter budget -- all four architectures in one bracket")
    sizes = {
        "mlp_pool_skip": count_params(mlp.build(data, **mlp.CONFIGS["mlp_pool_skip"])),
        "lstm_skip": count_params(L.build(data, **L.CONFIGS["lstm_skip"])),
        "transformer_skip": count_params(T.build(data, **T.CONFIGS["transformer_skip"])),
        "mlp_flat_skip": count_params(mlp.build(data, **mlp.CONFIGS["mlp_flat_skip"])),
    }
    for k, v in sizes.items():
        print(f"     {k:20s} {v:>9,d}")
    core = [sizes["mlp_pool_skip"], sizes["lstm_skip"], sizes["transformer_skip"]]
    check("the three compared architectures are within 2x of each other",
          max(core) / min(core) < 2.0, f"ratio {max(core)/min(core):.2f}")

    print("\n2. output plumbing is identical across architectures")
    set_seed(0)
    m = L.build(data, **L.CONFIGS["lstm_skip"]).eval()
    out = m(*blocks)
    check("output is (B, 24, 2, 7)", tuple(out.shape) == (64, 24, 2, 7), str(tuple(out.shape)))
    check("head is shared across hours (d_hidden -> 14)",
          m.head.out_features == 14 and m.head.in_features == m.d_hidden)
    check("head size matches the transformer's", m.head.out_features ==
          T.build(data, **T.CONFIGS["transformer"]).head.out_features)

    print("\n3. loss, monotonicity, skip")
    tl = float(PinballLoss()(out, y).detach())
    nl = metrics.pinball(y.numpy(), out.detach().numpy())
    check("torch pinball == numpy pinball",
          abs(tl - nl) / max(abs(nl), 1e-9) < 1e-5)
    d = np.diff(out.detach().numpy(), axis=-1)
    check("quantiles monotone", (d >= -1e-6).all(), f"min diff {d.min():.2e}")
    set_seed(0); a = L.build(data, skip=False).eval()(*blocks)
    set_seed(0); b = L.build(data, skip=True).eval()(*blocks)
    delta = float((a - b).abs().max().detach())
    check("skip barely moves the untrained output", delta < 0.1, f"max delta {delta:.4f}")

    print("\n4. recurrence is actually doing something")
    # Reversing the history must change the answer. If it does not, the encoder
    # LSTMs are ignoring order and the model is an expensive pooling layer.
    rev = list(blocks)
    rev[0] = blocks[0].flip(1)
    check("reversing history changes the output",
          not torch.allclose(m(*blocks), m(*rev), atol=1e-4))

    print("\n5. gradients flow -- overfit 32 samples")
    set_seed(0)
    m2 = L.build(data, dropout=0.0, skip=False)
    opt = torch.optim.Adam(m2.parameters(), lr=1e-2)
    lossf = PinballLoss()
    small = [b[:32] for b in blocks]
    ys = y[:32]
    first = float(lossf(m2(*small), ys).detach())
    for _ in range(300):
        opt.zero_grad(); l = lossf(m2(*small), ys); l.backward()
        torch.nn.utils.clip_grad_norm_(m2.parameters(), 5.0)
        opt.step()
    last = float(l.detach())
    check("loss falls by >70% overfitting a tiny batch", last < 0.3 * first,
          f"{first:.4f} -> {last:.4f}")

    print("\n6. end-to-end fit and the final timing number")
    tr, va, te = data.windows("2025-01-01", policy="rolling", years=2)
    t0 = time.time()
    tm = TorchModel(lambda: L.build(data, **L.CONFIGS["lstm_skip"]), "lstm_skip", seed=0).fit(data, tr, va)
    secs = time.time() - t0
    pred = tm.predict(data, te)
    s = metrics.summary(data.Y[te], pred)
    print(f"     lstm_skip        {secs:6.1f}s  {tm.epochs_run:3d} epochs  "
          f"{tm.n_params:>8,d} params  pinball {s['pinball']:.3f}  "
          f"DE-FR MAE {s['spread_DE_FR_mae']:.2f}")
    check("lstm_skip predicts the right shape",
          pred.shape == (len(te), 24, 2, 7), str(pred.shape))

    print("\n     2025-Q1 rolling scoreboard so far (DE-FR MAE / pinball):")
    print("       LEAR             13.81 / 4.320")
    print("       transformer_skip 14.91 / 4.845")
    print("       mlp_pool_skip    15.58 / 4.761")
    print(f"       lstm_skip        {s['spread_DE_FR_mae']:.2f} / {s['pinball']:.3f}")
    print("       ridge            17.40 / 5.152")
    print(f"\n     grid cost for lstm: 46 fits x 8 seeds x 2 configs = {2*46*8*secs/3600:.1f} h")

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED -- ready for the full grid")


if __name__ == "__main__":
    main()
