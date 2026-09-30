"""The LSTM network (PAPER.md, Section 4.3). Code identifier: lstm.
The linear-skip variant in CONFIGS was explored earlier and is not reported.

Encoder-decoder, which is what makes the comparison clean:

    encoder   one LSTM per history stream, run over 168 hours; take the final
              hidden state as that stream's summary
    bridge    the two summaries are projected to the initial hidden state of
              the decoder
    decoder   an LSTM run over the 24 target hours, reading each hour's
              known-ahead covariates, emitting one state per hour
    head      shared across hours, d -> 14 (2 spreads x 7 quantiles)

Using the decoder's per-hour states rather than flattening a single final state
keeps the head at 686 parameters, exactly as in the transformer -- so the three
architectures differ in how they process the sequence and NOT in how they emit
the answer. Any difference in results is then attributable to the mechanism under
test rather than to output plumbing.

`nn.LSTM` is used directly. There is no pedagogical reason to hand-roll a cell:
attention is the thing this project exists to understand, and the LSTM is here as
a control for "does recurrence alone explain whatever attention gains".
"""
import torch
import torch.nn as nn

from nn_common import (HORIZON, N_Q, N_TARGETS, OUT_WIDTH, LinearSkip,
                       StepEncoder, assemble, count_params, pool_sequence)


class SpreadLSTM(nn.Module):
    def __init__(self, n_auction, n_realised, n_ahead, n_calendar,
                 d_embed=16, d_hidden=48, n_layers=1, dropout=0.1,
                 skip=True, skip_rank=8, out_mode="quantile"):
        super().__init__()
        self.d_hidden, self.n_layers = d_hidden, n_layers

        # Separate encoders and separate LSTMs: the streams end at different
        # clock times and must not be merged on the feature axis (see nn_common).
        self.enc_auction = StepEncoder(n_auction, d_embed)
        self.enc_realised = StepEncoder(n_realised, d_embed)
        self.enc_target = StepEncoder(n_ahead + n_calendar, d_embed)

        kw = dict(batch_first=True, num_layers=n_layers,
                  dropout=dropout if n_layers > 1 else 0.0)
        self.lstm_auction = nn.LSTM(d_embed, d_hidden, **kw)
        self.lstm_realised = nn.LSTM(d_embed, d_hidden, **kw)
        self.lstm_target = nn.LSTM(d_embed, d_hidden, **kw)

        self.bridge_h = nn.Linear(2 * d_hidden, n_layers * d_hidden)
        self.bridge_c = nn.Linear(2 * d_hidden, n_layers * d_hidden)

        self.drop = nn.Dropout(dropout)
        self.out_mode = out_mode
        self.head = nn.Linear(d_hidden, N_TARGETS * OUT_WIDTH[out_mode])

        self.n_skip_in = (5 * (n_auction + n_realised)
                          + HORIZON * (n_ahead + n_calendar))
        self.skip = LinearSkip(self.n_skip_in, rank=skip_rank,
                               out_shape=(HORIZON, N_TARGETS, OUT_WIDTH[out_mode])) if skip else None

    def forward(self, auction, realised, ahead, calendar):
        B = auction.shape[0]
        _, (ha, _) = self.lstm_auction(self.enc_auction(auction))
        _, (hr, _) = self.lstm_realised(self.enc_realised(realised))
        summary = torch.cat([ha[-1], hr[-1]], dim=1)                # (B, 2d)

        h0 = self.bridge_h(summary).view(B, self.n_layers, self.d_hidden).transpose(0, 1)
        c0 = self.bridge_c(summary).view(B, self.n_layers, self.d_hidden).transpose(0, 1)

        tgt = self.enc_target(torch.cat([ahead, calendar], dim=-1))
        seq, _ = self.lstm_target(tgt, (h0.contiguous(), c0.contiguous()))
        out = self.head(self.drop(seq)).view(-1, HORIZON, N_TARGETS, OUT_WIDTH[self.out_mode])

        if self.skip is not None:
            raw = torch.cat([pool_sequence(auction), pool_sequence(realised),
                             ahead.flatten(1), calendar.flatten(1)], dim=1)
            out = out + self.skip(raw)

        return assemble(out, self.out_mode)


def build(data, d_embed=16, d_hidden=48, n_layers=1, dropout=0.1,
          skip=True, skip_rank=8, out_mode="quantile"):
    return SpreadLSTM(
        n_auction=len(data.cols["auction"]),
        n_realised=len(data.cols["realised"]),
        n_ahead=len(data.cols["ahead"]),
        n_calendar=len(data.cols["calendar"]),
        d_embed=d_embed, d_hidden=d_hidden, n_layers=n_layers,
        dropout=dropout, skip=skip, skip_rank=skip_rank, out_mode=out_mode,
    )


CONFIGS = {
    "lstm":       dict(skip=False),
    "lstm_skip":  dict(skip=True),
}

if __name__ == "__main__":
    import os, sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(ROOT, "harness"))
    from dataset import SpreadData
    d = SpreadData(os.path.join(ROOT, "data"))
    for name, cfg in CONFIGS.items():
        print(f"{name:14s} {count_params(build(d, **cfg)):>9,d} params")
