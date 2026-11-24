"""Architecture-only RL comparison with frozen attention and shared action scoring.

Feature construction accepts fit data only. Both arms retain the audited
two-phase Double-DQN trainer, rewards, masks, replay and finite-horizon targets.
The new arm changes the action representation as well as weight sharing; it is
not an equal-information or equal-parameter-count architecture ablation.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch
from torch import nn

from .stability import SwapTrainer
from .swap_rl import SwapContext


def band_features(X_fit, y_fit, A):
    X, y, A = [np.asarray(v, dtype=np.float64) for v in (X_fit, y_fit, A)]
    if X.ndim != 2 or y.ndim != 2 or A.shape != (y.shape[1], X.shape[1]):
        raise ValueError("Fit features, targets and frozen attention do not align")
    if len(X) != len(y) or not all(np.isfinite(v).all() for v in (X, y, A)):
        raise ValueError("Expected aligned finite fit data")
    def z(v):
        return (v-v.mean(0))/np.maximum(v.std(0, ddof=0), 1e-12)
    correlation = np.clip(z(X).T @ z(y)/len(X), -1., 1.)
    # Position is a column coordinate, never a wavelength in nm.
    position = np.linspace(-1., 1., X.shape[1])[:, None]
    return np.asarray(np.column_stack([position, X.shape[1]*A.T, correlation]), dtype=np.float32)


class SharedActionQNetwork(nn.Module):
    """One global state encoder, one candidate encoder and one shared scorer."""
    def __init__(self, state_dim, n_bands, features, hidden=64):
        super().__init__()
        features = torch.as_tensor(np.array(features, copy=True), dtype=torch.float32)
        if state_dim != 4*n_bands+3 or features.ndim != 2 or len(features) != n_bands:
            raise ValueError("Invalid action/state dimensions")
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite candidate features")
        self.n_bands, self.hidden = n_bands, hidden
        self.register_buffer("band_features", features)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.action_encoder = nn.Linear(features.shape[1]+4, hidden)
        self.action_bias = nn.Linear(hidden, 1)
        self.state_bias = nn.Linear(hidden, 1)

    def forward(self, state):
        batch = state.shape[0]
        local = state[:, :4*self.n_bands].reshape(batch, 4, self.n_bands).transpose(1, 2)
        static = self.band_features[None].expand(batch, -1, -1)
        keys = torch.relu(self.action_encoder(torch.cat([static, local], dim=-1)))
        query = self.state_encoder(state)
        return ((keys*query[:, None]).sum(-1)/math.sqrt(self.hidden)
                + self.action_bias(keys).squeeze(-1) + self.state_bias(query))


class AttentionArchitectureTrainer(SwapTrainer):
    """Extend through composition points without changing the legacy trainer."""
    def __init__(self, *args, architecture, A, R, features, probe_pool, **kwargs):
        if architecture not in {"vector_head", "shared_action"}:
            raise ValueError(architecture)
        super().__init__(*args, **kwargs)
        self.architecture, self.external_probe_pool = architecture, probe_pool
        if architecture == "shared_action":
            # The architectures cannot have identical tensors. Their seed,
            # initialization convention and training random streams are fixed.
            torch.manual_seed(self.seed)
            self.online = SharedActionQNetwork(4*self.ev.n_bands+3, self.ev.n_bands, features)
            self.target = SharedActionQNetwork(4*self.ev.n_bands+3, self.ev.n_bands, features)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
            self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config["learning_rate"])
        self.initial_state_hash = hashlib.sha256(b"".join(
            t.detach().numpy().tobytes() for t in self.online.state_dict().values())).hexdigest()
        self.context = SwapContext(self.ev, hint_type="frozen_attention",
            correlation=np.zeros(self.ev.n_bands), A=A, R=R, horizon=self.horizon)
        self.probe_context = SwapContext(self.pev, hint_type="frozen_attention",
            correlation=np.zeros(self.ev.n_bands), A=A, R=R, horizon=self.horizon)

    def probe(self, checkpoint):
        training_pool = self.pool
        try:
            self.pool = self.external_probe_pool
            result = super().probe(checkpoint)
        finally:
            self.pool = training_pool
        # Newly generated probe roots are paid queries too; include them in
        # the search incumbent even if the subsequent policy only worsens them.
        self.best_probe, self.best_probe_error = min(self.pev._cache.items(), key=lambda pair: pair[1])
        result["best_search_nmse"] = min(self.best_train_error, self.best_probe_error)
        return result
