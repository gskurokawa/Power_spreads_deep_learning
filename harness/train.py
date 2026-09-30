"""Generic torch training loop, wrapped to match the baseline interface.

Any torch model exposing forward(auction, realised, ahead, calendar) can be run
by the walk-forward runner through TorchModel, with no change to the runner.
"""
import copy
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models"))
from nn_common import PinballLoss, set_seed, count_params  # noqa: E402

BLOCKS = ("X_auction", "X_realised", "X_ahead", "X_calendar")


class Standardiser:
    """Per-feature mean/std, fitted on the training window only.

    Statistics are computed over (samples, timesteps) so one feature has one
    scale regardless of position in the window -- refitting per retrain point,
    which matters under drift: 2019 price levels and 2023 price levels are not
    the same variable in practice.
    """

    def fit(self, arrays):
        self.mu, self.sd = [], []
        for a in arrays:
            m = a.mean(axis=(0, 1), keepdims=True)
            s = a.std(axis=(0, 1), keepdims=True)
            self.mu.append(m)
            self.sd.append(np.where(s > 1e-8, s, 1.0))
        return self

    def transform(self, arrays):
        return [(a - m) / s for a, m, s in zip(arrays, self.mu, self.sd)]


class TorchModel:
    """fit(data, train_idx, val_idx) / predict(data, idx), like the baselines."""

    def __init__(self, builder, name, max_epochs=300, patience=25, lr=1e-3,
                 batch_size=64, weight_decay=1e-4, seed=0, device="cpu", verbose=False,
                 loss_module=None, predict_transform=None):
        """loss_module / predict_transform default to None: pinball loss, and the
        network's quantile output returned unchanged. Both are hooks for
        alternative losses or output transforms; the paper uses the defaults."""
        self.builder, self.name = builder, name
        self.loss_module = loss_module
        self.predict_transform = predict_transform
        self.max_epochs, self.patience = max_epochs, patience
        self.lr, self.batch_size, self.weight_decay = lr, batch_size, weight_decay
        self.seed, self.device, self.verbose = seed, device, verbose

    # ------------------------------------------------------------------ data
    def _blocks(self, data, idx):
        return [getattr(data, b.replace("X_", "X_"))[idx] for b in
                ("X_auction", "X_realised", "X_ahead", "X_calendar")]

    def _tensors(self, arrays):
        return [torch.tensor(a, dtype=torch.float32, device=self.device) for a in arrays]

    # ------------------------------------------------------------------- fit
    def fit(self, data, train_idx, val_idx):
        set_seed(self.seed)
        tr = self._blocks(data, train_idx)
        self.std = Standardiser().fit(tr)
        Xtr = self._tensors(self.std.transform(tr))
        ytr = torch.tensor(data.Y[train_idx], dtype=torch.float32, device=self.device)

        use_val = len(val_idx) >= 20
        if use_val:
            Xva = self._tensors(self.std.transform(self._blocks(data, val_idx)))
            yva = torch.tensor(data.Y[val_idx], dtype=torch.float32, device=self.device)

        self.model = self.builder().to(self.device)
        self.n_params = count_params(self.model)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        lossf = (self.loss_module() if self.loss_module else PinballLoss()).to(self.device)

        loader = DataLoader(TensorDataset(*Xtr, ytr), batch_size=self.batch_size,
                            shuffle=True, drop_last=False)

        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.model.train()
            for *xb, yb in loader:
                opt.zero_grad()
                loss = lossf(self.model(*xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

            if not use_val:
                continue
            self.model.eval()
            with torch.no_grad():
                v = float(lossf(self.model(*Xva), yva))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and epoch % 20 == 0:
                print(f"    epoch {epoch:3d}  val pinball {v:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val = best if use_val else np.nan
        self.epochs_run = epoch + 1
        return self

    # --------------------------------------------------------------- predict
    def predict(self, data, idx):
        self.model.eval()
        X = self._tensors(self.std.transform(self._blocks(data, idx)))
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                out.append(self.model(*[x[i:i + 512] for x in X]).cpu().numpy())
        pred = np.concatenate(out, axis=0)
        return self.predict_transform(pred) if self.predict_transform else pred
