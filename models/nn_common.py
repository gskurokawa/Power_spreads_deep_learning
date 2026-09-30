"""Shared pieces for the torch models (MLP, LSTM, transformer).

IMPORTANT -- the two history streams are NOT time-aligned.

`X_auction` runs to 23:00 on day D; `X_realised` stops at 09:00 on day D, because
metered outturn publishes with a lag (see data/feature_classes.py).
Both arrays are 168 rows long, so concatenating them on the feature axis would
silently pair 23:00 with 09:00 and quietly corrupt every sequence model. They are
therefore kept as SEPARATE streams everywhere, with their own encoders and, in the
transformer, their own token-type embedding.
"""
import numpy as np
import torch
import torch.nn as nn

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
N_Q = len(QUANTILES)
HORIZON = 24
N_TARGETS = 2

# Output width per spread, per hour: 7 monotone quantiles of the spread.
OUT_WIDTH = {"quantile": N_Q}


def assemble(raw, mode="quantile"):
    """Turn a network's raw output into its predictive form.

    Monotonicity is enforced by construction -- the lowest quantile plus softplus
    increments -- so quantile crossing is impossible rather than merely penalised.
    """
    if mode == "quantile":
        lo, inc = raw[..., :1], torch.nn.functional.softplus(raw[..., 1:])
        return torch.cat([lo, lo + torch.cumsum(inc, dim=-1)], dim=-1)
    raise ValueError(mode)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PinballLoss(nn.Module):
    """Mean pinball loss. pred (B, 24, 2, Q), target (B, 24, 2).

    Must agree with harness/metrics.pinball to ~1e-6; tests/test_torch_models.py
    checks that against the numpy implementation.
    """

    def __init__(self, quantiles=QUANTILES):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles).view(1, 1, 1, -1))

    def forward(self, pred, target):
        e = target.unsqueeze(-1) - pred
        return torch.maximum(self.q * e, (self.q - 1) * e).mean()


class StepEncoder(nn.Module):
    """Project one timestep's features to d, shared across all timesteps."""

    def __init__(self, n_features, d):
        super().__init__()
        self.lin = nn.Linear(n_features, d)

    def forward(self, x):                      # (B, T, F) -> (B, T, d)
        return self.lin(x)


class LinearSkip(nn.Module):
    """Low-rank linear path from the input straight to the output.

    El Mahtouta & Ziel (2026): let the linear part be captured linearly so the
    nonlinear body only has to learn the residual. It gives the ablation that
    sharpens the research question -- does attention add anything ONCE the linear
    component is already handled?

    WHY LOW RANK. A dense map from the raw flattened input (6,072 features) to the
    full output (24 x 2 x 7 = 336) costs 2.04M parameters -- 46x the rest of the
    pooled MLP, which would make every "with skip" model a linear model with a
    small neural ornament attached, and the ablation meaningless. Factoring it
    through a rank-r bottleneck costs ~8k instead. The ridge baseline already
    provides the full-rank linear map as a standalone model, so nothing is lost.
    """

    def __init__(self, n_in, rank=8, out_shape=(HORIZON, N_TARGETS, N_Q)):
        super().__init__()
        self.out_shape = out_shape
        n_out = int(np.prod(out_shape))
        self.down = nn.Linear(n_in, rank, bias=False)
        self.up = nn.Linear(rank, n_out)
        nn.init.zeros_(self.up.bias)
        nn.init.normal_(self.up.weight, std=1e-3)    # start as a no-op
        nn.init.normal_(self.down.weight, std=1.0 / np.sqrt(n_in))

    def forward(self, x_flat):
        return self.up(self.down(x_flat)).view(-1, *self.out_shape)


def pool_sequence(x):
    """mean / std / min / max / last over the time axis -> (B, 5*F).

    The compact history representation. Flattening 168 timesteps gives the MLP a
    first layer of ~185k parameters, several times the transformer's whole budget;
    pooling keeps the comparison closer to parameter-matched. Both are run.
    """
    return torch.cat([x.mean(1), x.std(1), x.amin(1), x.amax(1), x[:, -1, :]], dim=1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
