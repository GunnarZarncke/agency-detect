from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from amortized_agency.worlds import Episode


class ChannelEncoder1D(nn.Module):
    """Shared per-channel encoder; no variable-id embedding (cross-world generalization)."""

    def __init__(self, window: int, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W]
        h = self.conv(x.unsqueeze(1)).mean(dim=-1)  # [B, 32]
        return self.proj(h)


class SiameseAffinityModel(nn.Module):
    def __init__(self, window: int, embed_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.encoder = ChannelEncoder1D(window, hidden=hidden, out_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, W] -> [B, N, D]
        b, n, w = x.shape
        flat = x.reshape(b * n, w)
        emb = self.encoder(flat).reshape(b, n, -1)
        return emb

    def pair_logits(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        z = torch.cat([e1, e2, torch.abs(e1 - e2)], dim=-1)
        return self.head(z).squeeze(-1)

    def forward_pair(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.pair_logits(self.encoder(x1), self.encoder(x2)))

    @torch.no_grad()
    def affinity_matrix(self, window: np.ndarray, device: torch.device, batch_pairs: int = 512) -> np.ndarray:
        """Build [N,N] same-agent affinity from pairwise scores."""
        x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)  # [1, N, W]
        n = x.shape[1]
        emb = self.encode_channels(x)[0]  # [N, D]
        aff = np.eye(n, dtype=np.float32)
        pairs_i, pairs_j = [], []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_i.append(i)
                pairs_j.append(j)
        if not pairs_i:
            return aff
        for start in range(0, len(pairs_i), batch_pairs):
            pi = pairs_i[start : start + batch_pairs]
            pj = pairs_j[start : start + batch_pairs]
            e1 = emb[pi]
            e2 = emb[pj]
            scores = torch.sigmoid(self.pair_logits(e1, e2)).cpu().numpy()
            for a, b, s in zip(pi, pj, scores):
                aff[a, b] = aff[b, a] = float(s)
        return aff


def sample_pairs(episode: Episode, n_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x1, x2 [P,W], labels [P] from one episode."""
    w = episode.window  # [T, N]
    ids = episode.agent_ids
    n = w.shape[1]
    x1, x2, labels = [], [], []
    for _ in range(n_pairs):
        i = int(rng.integers(0, n))
        if rng.random() < 0.5:
            same = np.where(ids == ids[i])[0]
            j = int(rng.choice(same))
            y = 1.0
        else:
            diff = np.where(ids != ids[i])[0]
            if len(diff) == 0:
                continue
            j = int(rng.choice(diff))
            y = 0.0
        x1.append(w[:, i])
        x2.append(w[:, j])
        labels.append(y)
    return (
        np.stack(x1, axis=0).astype(np.float32),
        np.stack(x2, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def train_siamese(
    model: SiameseAffinityModel,
    episodes: List[Episode],
    *,
    epochs: int,
    pairs_per_episode: int,
    lr: float,
    device: torch.device,
    seed: int = 0,
) -> List[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses: List[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        rng.shuffle(episodes)
        for ep in episodes:
            x1, x2, y = sample_pairs(ep, pairs_per_episode, rng)
            if len(y) == 0:
                continue
            t1 = torch.from_numpy(x1).to(device)
            t2 = torch.from_numpy(x2).to(device)
            ty = torch.from_numpy(y).to(device)
            pred = model.forward_pair(t1, t2)
            loss = F.binary_cross_entropy(pred, ty)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses
