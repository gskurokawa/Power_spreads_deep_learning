"""The transformer (PAPER.md, Section 4.3). Code identifier: transformer.
Attention is written by hand. The linear-skip variant in CONFIGS was explored
earlier and is not reported.

TOKENISATION. Not one token per hour. The unit of this problem is the day: the
auction clears once daily for 24 hours at once, so the sequence the model should
reason over is days of history, not 168 undifferentiated hours. Three token types:

    7  auction-day tokens    each = 24h x 22 auction features, projected to d
    7  realised-day tokens   each = 24h x 11 realised features, projected to d
    24 target-hour tokens    each = that hour's known-ahead covariates + calendar

38 tokens, full self-attention. This is PatchTST's patching idea with the patch
length set by the market's own period rather than chosen by search.

The target-hour tokens are the reason the output head is small: rather than
flattening the encoder output into a 336-wide layer (150k parameters), each
target-hour token is projected to its own 14 values (2 spreads x 7 quantiles)
through a head shared across hours -- 686 parameters. Those tokens are not empty
placeholders; they are initialised from real per-hour features (load forecast,
wind/solar forecast, calendar) that are expected to be public before gate closure
(see data/feature_classes.py).

The auction and realised streams are NOT time-aligned (auction runs to 23:00 on
day D, realised stops at 09:00) so they are separate token types with separate
projections and a learned type embedding, never concatenated. See nn_common.
"""
import math

import torch
import torch.nn as nn

from nn_common import (HORIZON, N_Q, N_TARGETS, OUT_WIDTH, LinearSkip,
                       assemble, count_params, pool_sequence)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention, written out.

    Deliberately not nn.MultiheadAttention -- learning this is the point of the
    project. tests/test_transformer.py copies weights into torch's version and
    asserts the outputs agree.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must divide by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn = None                  # kept for inspecting attention maps

    def forward(self, x, store_attn=False):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # (B, T, D) -> (B, H, T, dh): split the model dimension across heads
        q = self.w_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.w_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.w_v(x).view(B, T, H, dh).transpose(1, 2)

        # scaled dot product; scale by sqrt(d_head) so the softmax does not
        # saturate as the head dimension grows
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)     # (B, H, T, T)
        attn = torch.softmax(scores, dim=-1)
        if store_attn:
            self.last_attn = attn.detach()
        out = self.drop(attn) @ v                              # (B, H, T, dh)

        out = out.transpose(1, 2).contiguous().view(B, T, D)   # concat heads
        return self.w_o(out)


class EncoderLayer(nn.Module):
    """Pre-norm transformer block.

    Pre-norm (LayerNorm before the sublayer, residual around it) rather than the
    original post-norm: it trains without a warmup schedule, which matters when
    there are ~700 training days and no budget for schedule tuning.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, store_attn=False):
        x = x + self.drop(self.attn(self.ln1(x), store_attn))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class SpreadTransformer(nn.Module):
    def __init__(self, n_auction, n_realised, n_ahead, n_calendar,
                 hist_len=168, d_model=48, n_heads=4, n_layers=2, d_ff=96,
                 dropout=0.1, skip=True, skip_rank=8, out_mode="quantile"):
        super().__init__()
        self.n_days = hist_len // HORIZON
        self.d_model = d_model

        self.proj_auction = nn.Linear(HORIZON * n_auction, d_model)
        self.proj_realised = nn.Linear(HORIZON * n_realised, d_model)
        self.proj_target = nn.Linear(n_ahead + n_calendar, d_model)

        # Learned embeddings, not sinusoidal: the structure here is calendar
        # (which day of history, which hour of the delivery day), not abstract
        # sequence position.
        self.type_emb = nn.Embedding(3, d_model)
        self.day_emb = nn.Embedding(self.n_days, d_model)
        self.hour_emb = nn.Embedding(HORIZON, d_model)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.out_mode = out_mode
        self.head = nn.Linear(d_model, N_TARGETS * OUT_WIDTH[out_mode])

        self.n_skip_in = (5 * (n_auction + n_realised)
                          + HORIZON * (n_ahead + n_calendar))
        self.skip = LinearSkip(self.n_skip_in, rank=skip_rank,
                               out_shape=(HORIZON, N_TARGETS, OUT_WIDTH[out_mode])) if skip else None

    def tokens(self, auction, realised, ahead, calendar):
        B = auction.shape[0]
        nd = self.n_days
        a = self.proj_auction(auction.view(B, nd, -1))
        r = self.proj_realised(realised.view(B, nd, -1))
        t = self.proj_target(torch.cat([ahead, calendar], dim=-1))

        day = torch.arange(nd, device=auction.device)
        hour = torch.arange(HORIZON, device=auction.device)
        a = a + self.day_emb(day) + self.type_emb(torch.tensor(0, device=a.device))
        r = r + self.day_emb(day) + self.type_emb(torch.tensor(1, device=a.device))
        t = t + self.hour_emb(hour) + self.type_emb(torch.tensor(2, device=a.device))
        return torch.cat([a, r, t], dim=1)                     # (B, 38, d)

    def forward(self, auction, realised, ahead, calendar, store_attn=False):
        x = self.tokens(auction, realised, ahead, calendar)
        for layer in self.layers:
            x = layer(x, store_attn)
        x = self.ln_out(x)

        tgt = x[:, -HORIZON:, :]                               # target-hour tokens
        out = self.head(tgt).view(-1, HORIZON, N_TARGETS, OUT_WIDTH[self.out_mode])

        if self.skip is not None:
            raw = torch.cat([pool_sequence(auction), pool_sequence(realised),
                             ahead.flatten(1), calendar.flatten(1)], dim=1)
            out = out + self.skip(raw)

        return assemble(out, self.out_mode)

    def attention_maps(self, auction, realised, ahead, calendar):
        """(n_layers, B, H, 38, 38) -- attention weights, for inspection."""
        self.eval()
        with torch.no_grad():
            self.forward(auction, realised, ahead, calendar, store_attn=True)
        return torch.stack([l.attn.last_attn for l in self.layers])


def build(data, d_model=48, n_heads=4, n_layers=2, d_ff=96, dropout=0.1,
          skip=True, skip_rank=8, out_mode="quantile"):
    return SpreadTransformer(
        n_auction=len(data.cols["auction"]),
        n_realised=len(data.cols["realised"]),
        n_ahead=len(data.cols["ahead"]),
        n_calendar=len(data.cols["calendar"]),
        hist_len=data.X_auction.shape[1],
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        dropout=dropout, skip=skip, skip_rank=skip_rank, out_mode=out_mode,
    )


CONFIGS = {
    "transformer":       dict(skip=False),
    "transformer_skip":  dict(skip=True),
}

if __name__ == "__main__":
    import os, sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(ROOT, "harness"))
    from dataset import SpreadData
    d = SpreadData(os.path.join(ROOT, "data"))
    for name, cfg in CONFIGS.items():
        print(f"{name:20s} {count_params(build(d, **cfg)):>9,d} params")
