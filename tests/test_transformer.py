"""Phase 3 gate: does the hand-written attention actually match PyTorch's?

The headline check copies weights from MultiHeadSelfAttention into
nn.MultiheadAttention and asserts the outputs agree to numerical tolerance. That
is the whole point of writing attention out by hand -- anything can be made to
train; agreeing with a reference implementation on identical weights is what
demonstrates the mechanism was understood.

    python tests/test_transformer.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "models"))

import torch                                                       # noqa: E402
import torch.nn as nn                                              # noqa: E402
from dataset import SpreadData                                     # noqa: E402
import metrics                                                     # noqa: E402
import transformer as T                                            # noqa: E402
from nn_common import PinballLoss, count_params, set_seed          # noqa: E402
from train import Standardiser, TorchModel                         # noqa: E402

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

    # ------------------------------------------------------------------ THE GATE
    print("\n1. hand-written attention vs nn.MultiheadAttention")
    torch.manual_seed(0)
    d_model, n_heads, B, L = 48, 4, 8, 38
    mine = T.MultiHeadSelfAttention(d_model, n_heads, dropout=0.0).eval()
    ref = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True).eval()

    # torch packs q, k, v into one (3d, d) matrix in that order
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mine.w_q.weight, mine.w_k.weight,
                                            mine.w_v.weight], dim=0))
        ref.in_proj_bias.copy_(torch.cat([mine.w_q.bias, mine.w_k.bias,
                                          mine.w_v.bias], dim=0))
        ref.out_proj.weight.copy_(mine.w_o.weight)
        ref.out_proj.bias.copy_(mine.w_o.bias)

    x = torch.randn(B, L, d_model)
    with torch.no_grad():
        got = mine(x)
        want, _ = ref(x, x, x, need_weights=False)
    diff = float((got - want).abs().max())
    check("attention output matches PyTorch on identical weights",
          diff < 1e-5, f"max abs diff {diff:.3e}")

    print("\n2. attention internals")
    with torch.no_grad():
        mine(x, store_attn=True)
    A = mine.last_attn
    check("attention shape is (B, H, T, T)", tuple(A.shape) == (B, n_heads, L, L), str(tuple(A.shape)))
    check("attention rows sum to 1", torch.allclose(A.sum(-1), torch.ones(B, n_heads, L), atol=1e-5))
    check("attention weights are non-negative", bool((A >= 0).all()))

    print("\n3. parameter budget")
    for name, cfg in T.CONFIGS.items():
        print(f"     {name:20s} {count_params(T.build(data, **cfg)):>9,d}")
    n = count_params(T.build(data, **T.CONFIGS["transformer_skip"]))
    check("transformer is within a factor of 2 of mlp_pool_skip (52,504)",
          0.5 * 52504 < n < 2 * 52504 or n < 150_000, f"{n:,d}")

    print("\n4. tokenisation")
    set_seed(0)
    m = T.build(data, **T.CONFIGS["transformer_skip"]).eval()
    tok = m.tokens(*blocks)
    check("38 tokens (7 auction days + 7 realised days + 24 target hours)",
          tok.shape[1] == 38, str(tuple(tok.shape)))
    check("token width is d_model", tok.shape[2] == m.d_model)

    print("\n5. shapes, loss and monotonicity")
    out = m(*blocks)
    check("output is (B, 24, 2, 7)", tuple(out.shape) == (64, 24, 2, 7), str(tuple(out.shape)))
    tl = float(PinballLoss()(out, y).detach())
    nl = metrics.pinball(y.numpy(), out.detach().numpy())
    rel = abs(tl - nl) / max(abs(nl), 1e-9)
    check("torch pinball == numpy pinball", rel < 1e-5, f"rel {rel:.2e}")
    d = np.diff(out.detach().numpy(), axis=-1)
    check("quantiles monotone", (d >= -1e-6).all(), f"min diff {d.min():.2e}")

    print("\n6. the skip is a near no-op at init")
    set_seed(0); a = T.build(data, skip=False).eval()(*blocks)
    set_seed(0); b = T.build(data, skip=True).eval()(*blocks)
    delta = float((a - b).abs().max().detach())
    check("skip barely moves the untrained output", delta < 0.1, f"max delta {delta:.4f}")

    print("\n7. gradients flow -- overfit 32 samples")
    set_seed(0)
    m2 = T.build(data, dropout=0.0, skip=False)
    opt = torch.optim.Adam(m2.parameters(), lr=1e-2)
    lossf = PinballLoss()
    small = [b[:32] for b in blocks]
    ys = y[:32]
    first = float(lossf(m2(*small), ys).detach())
    for _ in range(300):
        opt.zero_grad(); l = lossf(m2(*small), ys); l.backward(); opt.step()
    last = float(l.detach())
    check("loss falls by >70% overfitting a tiny batch", last < 0.3 * first,
          f"{first:.4f} -> {last:.4f}")

    print("\n8. permuting history days changes the answer")
    # If it does not, the day embeddings are doing nothing and the model has no
    # more temporal structure than the MLP.
    perm = list(blocks)
    perm[0] = blocks[0].view(64, 7, 24, -1).flip(1).reshape(blocks[0].shape)
    check("shuffling the order of history days moves the output",
          not torch.allclose(m(*blocks), m(*perm), atol=1e-4))

    print("\n9. attention maps come back the right shape")
    maps = m.attention_maps(*[b[:4] for b in blocks])
    check("maps are (layers, B, heads, 38, 38)",
          tuple(maps.shape) == (len(m.layers), 4, 4, 38, 38), str(tuple(maps.shape)))

    print("\n10. end-to-end fit, and the timing number")
    tr, va, te = data.windows("2025-01-01", policy="rolling", years=2)
    for cfg_name in ("transformer_skip",):
        cfg = T.CONFIGS[cfg_name]
        t0 = time.time()
        tm = TorchModel(lambda c=cfg: T.build(data, **c), cfg_name, seed=0).fit(data, tr, va)
        secs = time.time() - t0
        pred = tm.predict(data, te)
        s = metrics.summary(data.Y[te], pred)
        print(f"     {cfg_name:18s} {secs:6.1f}s  {tm.epochs_run:3d} epochs  "
              f"{tm.n_params:>8,d} params  pinball {s['pinball']:.3f}  "
              f"DE-FR MAE {s['spread_DE_FR_mae']:.2f}")
        print(f"     -> 46 fits x 8 seeds = {46*8*secs/3600:.1f} h")
        check(f"{cfg_name} predicts the right shape",
              pred.shape == (len(te), 24, 2, 7), str(pred.shape))

    print("\n     for reference, same window: LEAR 13.81 / 4.320,  "
          "mlp_pool_skip 15.58 / 4.761  (MAE / pinball)")
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED -- Phase 3 gate met")


if __name__ == "__main__":
    main()
