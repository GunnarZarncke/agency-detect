from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from amortized_agency.siamese import ChannelEncoder1D
from amortized_agency.worlds import Episode, same_agent_matrix


@dataclass
class SlotTrainConfig:
    """All loss terms share one global optimum: each agent in a distinct slot,
    each variable one-hot over slots. No term conflicts, so training is stable
    under arbitrarily many epochs (no early stopping needed)."""

    epochs: int = 25
    lr: float = 3e-4
    lambda_coassign: float = 1.0
    lambda_cohesion: float = 0.5
    lambda_contrast: float = 1.0
    lambda_sharp: float = 0.1
    contrast_temp: float = 0.2


class CrossChannelEncoder(nn.Module):
    """Context-aware channel encoder.

    Per-channel temporal conv keeps `n_tokens` time tokens (no premature pooling),
    then self-attention ACROSS channels at each time token lets co-varying channels
    exchange information. This exposes the relational (cross-channel correlation)
    structure that defines agent membership, which a per-channel encoder discards.
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
        self.n_tokens = n_tokens
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.token_proj = nn.Linear(conv_ch, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim, nhead=n_heads, dim_feedforward=enc_dim * 2, batch_first=True
        )
        self.channel_tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(enc_dim, enc_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, W] -> [B, N, enc_dim]
        b, n, w = x.shape
        h = self.conv(x.reshape(b * n, 1, w))  # [B*N, C, W]
        h = F.adaptive_avg_pool1d(h, self.n_tokens)  # [B*N, C, T']
        h = self.token_proj(h.transpose(1, 2))  # [B*N, T', enc_dim]
        t = h.shape[1]
        # Cross-channel attention at each time token: tokens=channels.
        h = h.reshape(b, n, t, -1).permute(0, 2, 1, 3).reshape(b * t, n, -1)  # [B*T', N, D]
        h = self.channel_tf(h)  # attend across channels
        h = h.reshape(b, t, n, -1).mean(dim=1)  # pool time -> [B, N, D]
        return self.out(h)


class SlotAttentionAffinity(nn.Module):
    """Canonical slot attention: slots COMPETE for each variable (softmax over slots).

    The clustering affinity is the slot co-assignment
        P[n, m] = <profile_n, profile_m>  in [0, 1],
    where profile_n is variable n's distribution over slots (sums to 1 over K).
    P is used identically as the BCE training target and the inference affinity,
    so there is no train/inference mismatch.
    """

    def __init__(
        self,
        window: int,
        num_slots: int = 16,
        enc_dim: int = 64,
        slot_dim: int = 64,
        slot_iters: int = 3,
        sample_slots: bool = True,
        cross_channel: bool = True,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.slot_iters = slot_iters
        self.enc_dim = enc_dim
        self.sample_slots = sample_slots
        if cross_channel:
            self.encoder = CrossChannelEncoder(window, enc_dim=enc_dim)
        else:
            self.encoder = ChannelEncoder1D(window, hidden=enc_dim, out_dim=enc_dim)
        self.cross_channel = cross_channel
        if sample_slots:
            # Shared Gaussian over slots -> exchangeable slots, resists collapse.
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim) - 1.0)
        else:
            self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
            self.register_parameter("slot_log_sigma", None)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(enc_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(enc_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, enc_dim),
            nn.GELU(),
            nn.Linear(enc_dim, slot_dim),
        )
        self.norm_inputs = nn.LayerNorm(enc_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

    def _init_slots(self, b: int, device: torch.device) -> torch.Tensor:
        if self.sample_slots:
            mu = self.slot_mu.expand(b, self.num_slots, self.slot_dim)
            sigma = self.slot_log_sigma.exp().expand(b, self.num_slots, self.slot_dim)
            return mu + sigma * torch.randn(b, self.num_slots, self.slot_dim, device=device)
        return self.slot_mu.expand(b, self.num_slots, self.slot_dim)

    def encode_variables(self, x: torch.Tensor) -> torch.Tensor:
        if self.cross_channel:
            emb = self.encoder(x)  # [B,N,enc_dim], context-aware
        else:
            b, n, w = x.shape
            emb = self.encoder(x.reshape(b * n, w)).reshape(b, n, -1)
        return self.norm_inputs(emb)

    def _slot_attention(self, encoded: torch.Tensor) -> torch.Tensor:
        b, n, _ = encoded.shape
        k = self.to_k(encoded)
        v = self.to_v(encoded)
        slots = self._init_slots(b, encoded.device)
        attn = None
        for _ in range(self.slot_iters):
            q = self.to_q(self.norm_slots(slots))
            logits = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.slot_dim)
            # Slots compete for each variable: softmax over slots (dim=1).
            attn = logits.softmax(dim=1) + 1e-8  # [B,K,N], sums to 1 over K
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean over N
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots.reshape(-1, self.slot_dim),
            ).reshape(b, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(slots)
        assert attn is not None
        return attn

    @staticmethod
    def profile_from_attn(attn: torch.Tensor) -> torch.Tensor:
        """Variable -> slot distribution [B,N,K] (sums to 1 over K)."""
        return attn.transpose(1, 2)

    @staticmethod
    def coassign(profile: torch.Tensor) -> torch.Tensor:
        """P[n,m] = <profile_n, profile_m> in [0,1]."""
        return torch.bmm(profile, profile.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_variables(x)
        attn = self._slot_attention(encoded)
        profile = self.profile_from_attn(attn)
        return profile, self.coassign(profile)

    @torch.no_grad()
    def affinity_matrix(
        self,
        window: np.ndarray,
        device: torch.device,
        eval_samples: int = 5,
    ) -> np.ndarray:
        x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
        reps = eval_samples if self.sample_slots else 1
        acc = None
        for _ in range(reps):
            _, p = self.forward(x)
            acc = p if acc is None else acc + p
        aff = (acc / reps)[0].cpu().numpy().astype(np.float32)
        np.fill_diagonal(aff, 1.0)
        return aff


def _coassign_bce(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Class-balanced BCE on the co-assignment matrix, excluding the diagonal.

    Reachable: same-agent vars -> identical one-hot profiles (P=1); different-agent
    vars -> orthogonal slots (P=0). Requires num_slots >= num_agents.
    """
    b, n, _ = prob.shape
    eye = torch.eye(n, device=prob.device, dtype=torch.bool).unsqueeze(0)
    pos = (target > 0.5) & ~eye
    neg = (target <= 0.5) & ~eye
    p = prob.clamp(1e-6, 1.0 - 1e-6)
    pos_loss = -torch.log(p[pos]).mean() if pos.any() else prob.new_tensor(0.0)
    neg_loss = -torch.log(1.0 - p[neg]).mean() if neg.any() else prob.new_tensor(0.0)
    return 0.5 * (pos_loss + neg_loss)


def _cohesion(profile: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
    """Same-agent variables should share a profile. Optimum agrees with BCE."""
    b = profile.shape[0]
    loss = profile.new_tensor(0.0)
    count = 0
    for bi in range(b):
        ids = agent_ids[bi]
        for a in torch.unique(ids):
            mask = ids == a
            if mask.sum() < 2:
                continue
            prof = profile[bi, mask, :]
            mean = prof.mean(dim=0, keepdim=True)
            loss = loss + F.mse_loss(prof, mean.expand_as(prof))
            count += 1
    return loss / max(count, 1)


def _supcon(profile: torch.Tensor, agent_ids: torch.Tensor, temp: float) -> torch.Tensor:
    """Vectorized supervised contrastive loss over slot-profiles (no per-variable loop)."""
    b = profile.shape[0]
    total = profile.new_tensor(0.0)
    count = 0
    for bi in range(b):
        z = F.normalize(profile[bi], dim=-1)  # [N,K]
        ids = agent_ids[bi]
        n = z.shape[0]
        sim = (z @ z.t()) / temp  # [N,N]
        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (ids[:, None] == ids[None, :]) & ~eye
        pos_counts = pos.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            continue
        pos_log_prob = (log_prob.masked_fill(~pos, 0.0).sum(dim=1))[valid] / pos_counts[valid]
        total = total + (-pos_log_prob.mean())
        count += 1
    return total / max(count, 1)


def _sharp(profile: torch.Tensor) -> torch.Tensor:
    """Each variable should commit to one slot: low entropy of profile over slots."""
    return -(profile * (profile + 1e-8).log()).sum(dim=-1).mean()


def _slot_episode_loss(
    model: SlotAttentionAffinity,
    episode: Episode,
    device: torch.device,
    cfg: SlotTrainConfig,
) -> torch.Tensor:
    x = torch.from_numpy(episode.window.T.astype(np.float32)).unsqueeze(0).to(device)
    agent_ids = torch.from_numpy(episode.agent_ids).unsqueeze(0).to(device)
    target = torch.from_numpy(same_agent_matrix(episode.agent_ids)).unsqueeze(0).to(device)

    profile, prob = model(x)

    l_coassign = _coassign_bce(prob, target)
    l_cohesion = _cohesion(profile, agent_ids)
    l_contrast = _supcon(profile, agent_ids, cfg.contrast_temp)
    l_sharp = _sharp(profile)

    return (
        cfg.lambda_coassign * l_coassign
        + cfg.lambda_cohesion * l_cohesion
        + cfg.lambda_contrast * l_contrast
        + cfg.lambda_sharp * l_sharp
    )


def train_slot(
    model: SlotAttentionAffinity,
    episodes: List[Episode],
    *,
    cfg: SlotTrainConfig,
    device: torch.device,
) -> List[float]:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses: List[float] = []
    model.train()
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        for ep in episodes:
            loss = _slot_episode_loss(model, ep, device, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(len(episodes), 1))
    return losses
