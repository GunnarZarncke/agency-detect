from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from amortized_agency.slot_model import CrossChannelEncoder, _coassign_bce
from amortized_agency.worlds import Episode, same_agent_matrix


@dataclass
class ContextTrainConfig:
    epochs: int = 40
    lr: float = 3e-4


class ContextualAffinityModel(nn.Module):
    """Cross-channel (context-aware) encoder + direct pairwise same-agent affinity.

    Unlike the slot model, there is no K-slot bottleneck: the affinity is the
    Gram matrix of contextual channel embeddings, trained with balanced BCE
    against the same-agent target. Diagnostics showed this is what the slot
    readout could not express.
    """

    def __init__(
        self,
        window: int,
        enc_dim: int = 64,
        conv_ch: int = 32,
        n_tokens: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = CrossChannelEncoder(
            window, enc_dim=enc_dim, conv_ch=conv_ch,
            n_tokens=n_tokens, n_heads=n_heads, n_layers=n_layers,
        )
        self.proj = nn.Linear(enc_dim, enc_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.encoder(x)), dim=-1)  # [B,N,D]

    def affinity(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        sim = torch.bmm(z, z.transpose(1, 2)).clamp(-1.0, 1.0)
        return (sim + 1.0) / 2.0  # [B,N,N] in [0,1]

    @torch.no_grad()
    def affinity_matrix(self, window: np.ndarray, device: torch.device) -> np.ndarray:
        x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
        aff = self.affinity(x)[0].cpu().numpy().astype(np.float32)
        np.fill_diagonal(aff, 1.0)
        return aff


def train_context(
    model: ContextualAffinityModel,
    episodes: List[Episode],
    *,
    cfg: ContextTrainConfig,
    device: torch.device,
) -> List[float]:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses: List[float] = []
    model.train()
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        for ep in episodes:
            x = torch.from_numpy(ep.window.T.astype(np.float32)).unsqueeze(0).to(device)
            target = torch.from_numpy(same_agent_matrix(ep.agent_ids)).unsqueeze(0).to(device)
            prob = model.affinity(x)
            loss = _coassign_bce(prob, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(len(episodes), 1))
    return losses
