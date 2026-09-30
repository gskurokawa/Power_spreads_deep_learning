"""The pooled MLP (PAPER.md, Section 4.3). Code identifier: mlp_pool.

Each history variable's 168 hourly values are reduced to five summary statistics
(mean, standard deviation, minimum, maximum, most recent value). The flattened-history
and linear-skip variants in CONFIGS were explored earlier and are not reported.

The MLP has no notion that its input is a sequence: shuffle the timesteps
consistently and it learns equally well. That is the point. It is the control
against which recurrence (LSTM) and attention (transformer) are measured.
"""
import torch
import torch.nn as nn

from nn_common import (HORIZON, N_Q, N_TARGETS, OUT_WIDTH, LinearSkip,
                       StepEncoder, assemble, count_params, pool_sequence)


class MLPNet(nn.Module):
    def __init__(self, n_auction, n_realised, n_ahead, n_calendar,
                 hist_len=168, d_embed=8, hidden=(64, 64), dropout=0.2,
                 history="pool", skip=True, skip_rank=8, out_mode="quantile"):
        super().__init__()
        if history not in ("pool", "flat"):
            raise ValueError(history)
        self.history = history

        # Separate encoders: the two history streams end at different clock times
        # and must never be concatenated on the feature axis (see nn_common).
        self.enc_auction = StepEncoder(n_auction, d_embed)
        self.enc_realised = StepEncoder(n_realised, d_embed)
        self.enc_ahead = StepEncoder(n_ahead + n_calendar, d_embed)

        if history == "flat":
            n_hist = 2 * hist_len * d_embed
        else:
            n_hist = 2 * 5 * d_embed                      # mean/std/min/max/last
        n_in = n_hist + HORIZON * d_embed

        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.out_mode = out_mode
        W = OUT_WIDTH[out_mode]
        layers.append(nn.Linear(prev, HORIZON * N_TARGETS * W))
        self.body = nn.Sequential(*layers)

        # The skip reads a POOLED view of the raw inputs, not the full 6,072-long
        # flatten: mean/std/min/max/last per raw feature per stream, plus the
        # known-ahead block. Keeps the linear path honest without letting it
        # swamp the parameter budget.
        self.n_skip_in = (5 * (n_auction + n_realised)
                          + HORIZON * (n_ahead + n_calendar))
        self.skip = LinearSkip(self.n_skip_in, rank=skip_rank,
                               out_shape=(HORIZON, N_TARGETS, OUT_WIDTH[out_mode])) if skip else None

    def forward(self, auction, realised, ahead, calendar):
        a = self.enc_auction(auction)
        r = self.enc_realised(realised)
        f = self.enc_ahead(torch.cat([ahead, calendar], dim=-1))

        if self.history == "flat":
            hist = torch.cat([a.flatten(1), r.flatten(1)], dim=1)
        else:
            hist = torch.cat([pool_sequence(a), pool_sequence(r)], dim=1)

        z = torch.cat([hist, f.flatten(1)], dim=1)
        out = self.body(z).view(-1, HORIZON, N_TARGETS, OUT_WIDTH[self.out_mode])

        if self.skip is not None:
            raw = torch.cat([pool_sequence(auction), pool_sequence(realised),
                             ahead.flatten(1), calendar.flatten(1)], dim=1)
            out = out + self.skip(raw)

        # Quantile crossing is meaningless: enforce monotonicity by construction
        # rather than hoping the loss discourages it. The model predicts the
        # lowest quantile plus non-negative increments.
        return assemble(out, self.out_mode)


def build(data, history="pool", skip=True, hidden=(64, 64), d_embed=8,
          dropout=0.2, skip_rank=8, out_mode="quantile"):
    return MLPNet(
        n_auction=len(data.cols["auction"]),
        n_realised=len(data.cols["realised"]),
        n_ahead=len(data.cols["ahead"]),
        n_calendar=len(data.cols["calendar"]),
        hist_len=data.X_auction.shape[1],
        d_embed=d_embed, hidden=hidden, dropout=dropout,
        history=history, skip=skip, skip_rank=skip_rank, out_mode=out_mode,
    )


CONFIGS = {
    "mlp_pool":           dict(history="pool", skip=False),
    "mlp_pool_skip":      dict(history="pool", skip=True),
    "mlp_flat":           dict(history="flat", skip=False),
    "mlp_flat_skip":      dict(history="flat", skip=True),
}

if __name__ == "__main__":       # parameter budget at a glance
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "harness"))
    from dataset import SpreadData
    d = SpreadData(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data"))
    for name, cfg in CONFIGS.items():
        print(f"{name:18s} {count_params(build(d, **cfg)):>9,d} params")
