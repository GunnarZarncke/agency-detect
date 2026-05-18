from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelConfig:
    num_vars: int
    window: int = 16
    num_slots: int = 16
    slot_dim: int = 16
    var_id_dim: int = 16
    enc_dim: int = 64
    hidden_dim: int = 128
    graph_msg_dim: int = 64
    horizon: int = 1
    slot_iters: int = 3
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: Optional[str] = None
    use_agency_regularizer: bool = False
    lambda_sparse: float = 1e-3
    lambda_control: float = 0.05
    lambda_memory: float = 0.05
    lambda_epsilon_blanket: float = 0.05
    epsilon_blanket: float = 0.05
    warmup_epochs: int = 5


@dataclass
class RefineConfig:
    """MI-partition-guided refinement after initial slot-attention training."""

    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: Optional[str] = None
    lambda_align: float = 2.0
    lambda_recon: float = 1.0
    lambda_pred: float = 1.0
    lambda_sparse: float = 1e-3
    mi_bins: int = 8
    mi_max_lag: int = 3
    freeze_var_encoder: bool = True
    freeze_dynamics: bool = True
    target_smoothing: float = 0.05


class WindowTraceDataset(Dataset):
    def __init__(self, trace: np.ndarray, window: int, horizon: int = 1, normalize: bool = True):
        if trace.ndim != 2:
            raise ValueError("trace must have shape [T, N]")
        if len(trace) <= window + horizon:
            raise ValueError("trace is too short for the requested window+horizon")

        x = trace.astype(np.float32)
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-6
        if normalize:
            x = (x - self.mean) / self.std

        self.x = torch.from_numpy(x)
        self.window = window
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.x) - self.window - self.horizon + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t0 = idx
        t1 = idx + self.window
        return {
            "x_window": self.x[t0:t1],
            "x_now": self.x[t1 - 1],
            "x_next": self.x[t1 + self.horizon - 1],
            "x_prev": self.x[t1 - 2] if self.window >= 2 else self.x[t1 - 1],
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VariableEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.var_embed = nn.Embedding(cfg.num_vars, cfg.var_id_dim)
        self.encoder = MLP(cfg.window + cfg.var_id_dim, cfg.hidden_dim, cfg.enc_dim, cfg.dropout)

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        # x_window: [B, W, N]
        b, w, n = x_window.shape
        if w != self.cfg.window or n != self.cfg.num_vars:
            raise ValueError(f"expected [B,{self.cfg.window},{self.cfg.num_vars}], got {tuple(x_window.shape)}")
        x_by_var = x_window.transpose(1, 2)  # [B, N, W]
        ids = torch.arange(n, device=x_window.device)
        emb = self.var_embed(ids).unsqueeze(0).expand(b, -1, -1)  # [B, N, D]
        return self.encoder(torch.cat([x_by_var, emb], dim=-1))  # [B, N, enc_dim]


class SlotAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.slot_mu = nn.Parameter(torch.randn(1, cfg.num_slots, cfg.slot_dim) * 0.02)
        self.to_q = nn.Linear(cfg.slot_dim, cfg.slot_dim, bias=False)
        self.to_k = nn.Linear(cfg.enc_dim, cfg.slot_dim, bias=False)
        self.to_v = nn.Linear(cfg.enc_dim, cfg.slot_dim, bias=False)
        self.gru = nn.GRUCell(cfg.slot_dim, cfg.slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(cfg.slot_dim),
            nn.Linear(cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.slot_dim),
        )
        self.norm_inputs = nn.LayerNorm(cfg.enc_dim)
        self.norm_slots = nn.LayerNorm(cfg.slot_dim)

    def forward(self, encoded_vars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoded_vars: [B, N, enc_dim]
        b, n, _ = encoded_vars.shape
        x = self.norm_inputs(encoded_vars)
        k = self.to_k(x)
        v = self.to_v(x)
        slots = self.slot_mu.expand(b, -1, -1)

        attn = None
        for _ in range(self.cfg.slot_iters):
            q = self.to_q(self.norm_slots(slots))
            logits = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.cfg.slot_dim)
            attn = logits.softmax(dim=1) + 1e-8       # slots compete for variables
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v)
            slots = self.gru(updates.reshape(-1, self.cfg.slot_dim), slots.reshape(-1, self.cfg.slot_dim))
            slots = slots.reshape(b, self.cfg.num_slots, self.cfg.slot_dim)
            slots = slots + self.mlp(slots)

        return slots, attn  # attn: [B, K, N], soft assignment of variables to slots


class SparseGraphGRU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.edge_logits = nn.Parameter(torch.randn(cfg.num_slots, cfg.num_slots) * 0.01)
        self.msg = nn.Sequential(
            nn.Linear(2 * cfg.slot_dim, cfg.graph_msg_dim),
            nn.GELU(),
            nn.Linear(cfg.graph_msg_dim, cfg.slot_dim),
        )
        self.gru = nn.GRUCell(cfg.slot_dim, cfg.slot_dim)
        self.self_update = nn.Linear(cfg.slot_dim, cfg.slot_dim)

    def adjacency(self) -> torch.Tensor:
        a = torch.sigmoid(self.edge_logits)
        eye = torch.eye(self.cfg.num_slots, device=a.device)
        return a * (1.0 - eye)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # slots: [B, K, D]
        b, k, d = slots.shape
        a = self.adjacency()  # [K, K], edge j->i represented as A[i,j]
        zi = slots.unsqueeze(2).expand(b, k, k, d)
        zj = slots.unsqueeze(1).expand(b, k, k, d)
        pair = torch.cat([zi, zj], dim=-1)
        messages = self.msg(pair) * a.view(1, k, k, 1)
        incoming = messages.sum(dim=2) / (a.sum(dim=1).view(1, k, 1) + 1e-6)
        incoming = incoming + self.self_update(slots)
        next_slots = self.gru(incoming.reshape(-1, d), slots.reshape(-1, d)).reshape(b, k, d)
        return next_slots, a


class SlotDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.to_var = nn.Sequential(
            nn.Linear(cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_vars),
        )
        self.slot_weight = nn.Linear(cfg.slot_dim, 1)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        # Mixture over slot-wise predictions.
        pred_per_slot = self.to_var(slots)              # [B, K, N]
        weights = self.slot_weight(slots).softmax(dim=1)  # [B, K, 1]
        return (weights * pred_per_slot).sum(dim=1)     # [B, N]


class AgencyLatentModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.var_encoder = VariableEncoder(cfg)
        self.slot_attention = SlotAttention(cfg)
        self.dynamics = SparseGraphGRU(cfg)
        self.decoder_now = SlotDecoder(cfg)
        self.decoder_next = SlotDecoder(cfg)
        self.action_head = nn.Sequential(
            nn.Linear(2 * cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.slot_dim),
        )
        self.episode_head = nn.Sequential(
            nn.Linear(cfg.num_slots * cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def encode(self, x_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.var_encoder(x_window)
        return self.slot_attention(encoded)

    def forward(self, x_window: torch.Tensor) -> Dict[str, torch.Tensor]:
        slots, assign = self.encode(x_window)
        pred_slots, adjacency = self.dynamics(slots)
        x_now_hat = self.decoder_now(slots)
        x_next_hat = self.decoder_next(pred_slots)
        action_like = self.action_head(torch.cat([slots, pred_slots - slots], dim=-1))
        episode_logit = self.episode_head(slots.flatten(start_dim=1)).squeeze(-1)
        return {
            "slots": slots,
            "pred_slots": pred_slots,
            "assign": assign,
            "adjacency": adjacency,
            "x_now_hat": x_now_hat,
            "x_next_hat": x_next_hat,
            "action_like": action_like,
            "episode_logit": episode_logit,
        }


class AgencyRegularizer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.future_from_action = nn.Sequential(
            nn.Linear(2 * cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.slot_dim),
        )
        self.next_from_prev = nn.Sequential(
            nn.Linear(2 * cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.slot_dim),
        )
        self.leak_discriminator = nn.Sequential(
            nn.Linear(3 * cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.slot_dim),
        )

    @staticmethod
    def _offdiag_roll(x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, shifts=1, dims=0)

    def control_score_loss(self, action_like: torch.Tensor, slots: torch.Tensor, pred_slots: torch.Tensor) -> torch.Tensor:
        # Contrastive proxy for I(a_i; z_j_next | z_j). Uses shuffled negatives.
        b, k, d = slots.shape
        target = pred_slots.detach()
        pred = self.future_from_action(torch.cat([action_like, slots], dim=-1))
        pos = F.cosine_similarity(pred, target, dim=-1)
        neg = F.cosine_similarity(pred, self._offdiag_roll(target), dim=-1)
        return -(pos - neg).mean()

    def memory_score_loss(self, slots: torch.Tensor, pred_slots: torch.Tensor, prev_slots: Optional[torch.Tensor]) -> torch.Tensor:
        # Rewards lagged slot state if it improves next-slot prediction.
        if prev_slots is None:
            return slots.new_tensor(0.0)
        pred = self.next_from_prev(torch.cat([slots, prev_slots], dim=-1))
        baseline = slots
        improvement = F.mse_loss(baseline, pred_slots.detach()) - F.mse_loss(pred, pred_slots.detach())
        return -improvement

    def epsilon_blanket_loss(self, slots: torch.Tensor, pred_slots: torch.Tensor, epsilon: float) -> torch.Tensor:
        # Cheap differentiable proxy: each slot tries to explain its own next state without
        # leaving strong unexplained residual dependence with the rest.
        # This is not a final UAD test; it is only a shaping loss.
        b, k, d = slots.shape
        others = (slots.sum(dim=1, keepdim=True) - slots) / max(k - 1, 1)
        pred_others_next = (pred_slots.sum(dim=1, keepdim=True) - pred_slots) / max(k - 1, 1)
        leak_pred = self.leak_discriminator(torch.cat([slots, pred_slots, others], dim=-1))
        residual = F.mse_loss(leak_pred, pred_others_next.detach(), reduction="none").mean(dim=-1)
        raw = F.mse_loss(others, pred_others_next.detach(), reduction="none").mean(dim=-1) + 1e-6
        normalized_leak = 1.0 - residual / raw
        violation = F.relu(normalized_leak - epsilon)
        return violation.pow(2).mean()

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        prev_slots: Optional[torch.Tensor],
        epsilon: float,
    ) -> Dict[str, torch.Tensor]:
        control = self.control_score_loss(out["action_like"], out["slots"], out["pred_slots"])
        memory = self.memory_score_loss(out["slots"], out["pred_slots"], prev_slots)
        blanket = self.epsilon_blanket_loss(out["slots"], out["pred_slots"], epsilon)
        return {"control": control, "memory": memory, "epsilon_blanket": blanket}


def discretize_trace_columns(trace: np.ndarray, bins: int = 8) -> np.ndarray:
    """Per-column quantile discretization for MI clustering."""
    if bins < 2:
        raise ValueError("bins must be >= 2")
    t_len, n_vars = trace.shape
    out = np.zeros((t_len, n_vars), dtype=np.int64)
    quantiles = np.linspace(0, 1, bins + 1)
    for j in range(n_vars):
        edges = np.quantile(trace[:, j], quantiles)
        edges = np.maximum.accumulate(edges)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        out[:, j] = np.clip(np.digitize(trace[:, j], edges[1:-1], right=False), 0, bins - 1)
    return out


def mi_cluster_variable_labels(
    trace: np.ndarray,
    num_clusters: int,
    bins: int = 8,
    max_lag: int = 3,
) -> np.ndarray:
    """
    Agglomerative clustering on lagged MI between raw variables.
    Returns integer labels [N] in 0..num_clusters-1 for active variables;
    inactive (zero variance) variables get label -1.
    """
    from agency_detect.detection import build_similarity_matrix
    from sklearn.cluster import AgglomerativeClustering

    disc = discretize_trace_columns(trace, bins=bins)
    n_vars = disc.shape[1]
    var_variance = trace.var(axis=0)
    active_idx = np.where(var_variance > 0.0)[0]
    if len(active_idx) < 2:
        return np.full(n_vars, -1, dtype=np.int64)

    data_active = disc[:, active_idx].astype(np.float64)
    _sim, dist = build_similarity_matrix(data_active, max_lag=max_lag)
    n_clust = min(num_clusters, len(active_idx))
    labels_active = AgglomerativeClustering(
        n_clusters=n_clust, metric="precomputed", linkage="complete"
    ).fit_predict(dist)

    labels = np.full(n_vars, -1, dtype=np.int64)
    for local_i, lbl in enumerate(labels_active):
        labels[int(active_idx[local_i])] = int(lbl)
    return labels


def match_mi_clusters_to_slots(
    labels: np.ndarray,
    avg_assign: np.ndarray,
) -> Dict[int, int]:
    """Hungarian match MI clusters to slots using overlap with mean assignment."""
    from scipy.optimize import linear_sum_assignment

    valid = labels >= 0
    if not np.any(valid):
        return {}

    clusters = np.unique(labels[valid])
    cluster_vecs = np.zeros((len(clusters), labels.shape[0]), dtype=np.float64)
    for i, c in enumerate(clusters):
        cluster_vecs[i, labels == c] = 1.0
        cluster_vecs[i] /= cluster_vecs[i].sum() + 1e-9

    k_slots = avg_assign.shape[0]
    cost = np.zeros((len(clusters), k_slots), dtype=np.float64)
    for i in range(len(clusters)):
        for k in range(k_slots):
            cost[i, k] = -float(np.dot(cluster_vecs[i], avg_assign[k]))

    row, col = linear_sum_assignment(cost)
    return {int(clusters[int(row[i])]): int(col[i]) for i in range(len(row))}


def build_mi_alignment_target(
    labels: np.ndarray,
    num_slots: int,
    cluster_to_slot: Dict[int, int],
    smoothing: float = 0.05,
) -> np.ndarray:
    """
    Soft slot assignment target [K, N] from MI partition.
    Each active variable assigns mass to its matched slot; unused slots get smoothing.
    """
    n_vars = labels.shape[0]
    target = np.full((num_slots, n_vars), smoothing / max(num_slots, 1), dtype=np.float32)
    for cluster_id, slot_id in cluster_to_slot.items():
        mask = labels == cluster_id
        if np.any(mask):
            target[slot_id, mask] = 1.0
    col_sum = target.sum(axis=0, keepdims=True)
    target = target / (col_sum + 1e-8)
    return target


def alignment_loss(assign: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(assign || target) for assign, target shaped [B, K, N] (softmax over slots)."""
    a = assign.clamp(min=1e-8)
    t = target.clamp(min=1e-8)
    return (a * (a.log() - t.log())).sum(dim=(1, 2)).mean()


@torch.no_grad()
def _mean_assignment(model: AgencyLatentModel, trace: np.ndarray, batch_size: int = 256) -> np.ndarray:
    latent = encode_trace(model, trace, batch_size=batch_size)
    return latent["assign"].mean(axis=0)


def refine_model_with_mi(
    model: AgencyLatentModel,
    trace: np.ndarray,
    num_agents: int,
    refine_cfg: Optional[RefineConfig] = None,
) -> Tuple[AgencyLatentModel, Dict[str, Any]]:
    """
    Refine a trained model so slot assignments align with an MI variable partition.

    Coarse-to-fine: lagged-MI clustering proposes agent-level groups; refinement
    fine-tunes (primarily slot attention) to match that partition while keeping
    reconstruction/prediction losses.
    """
    if refine_cfg is None:
        refine_cfg = RefineConfig()

    device = choose_device(refine_cfg.device)
    model = model.to(device)
    labels = mi_cluster_variable_labels(
        trace, num_clusters=num_agents, bins=refine_cfg.mi_bins, max_lag=refine_cfg.mi_max_lag
    )
    avg_assign = _mean_assignment(model, trace, batch_size=refine_cfg.batch_size)
    cluster_to_slot = match_mi_clusters_to_slots(labels, avg_assign)
    if not cluster_to_slot:
        raise RuntimeError("MI clustering produced no valid clusters for refinement")

    target_np = build_mi_alignment_target(
        labels, model.cfg.num_slots, cluster_to_slot, smoothing=refine_cfg.target_smoothing
    )
    target_t = torch.from_numpy(target_np).to(device)

    if refine_cfg.freeze_var_encoder:
        for p in model.var_encoder.parameters():
            p.requires_grad = False
    if refine_cfg.freeze_dynamics:
        for p in model.dynamics.parameters():
            p.requires_grad = False
        model.dynamics.edge_logits.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=refine_cfg.lr, weight_decay=refine_cfg.weight_decay)

    ds = WindowTraceDataset(trace, model.cfg.window, model.cfg.horizon)
    dl = DataLoader(ds, batch_size=refine_cfg.batch_size, shuffle=True, drop_last=True)

    history: Dict[str, list] = {
        "loss": [],
        "recon": [],
        "pred": [],
        "align": [],
        "sparse": [],
    }

    n_active = int(np.sum(labels >= 0))
    print(
        f"MI refine: {len(cluster_to_slot)} clusters -> slots, "
        f"{n_active}/{len(labels)} vars labeled, target smoothing={refine_cfg.target_smoothing}"
    )

    for epoch in range(refine_cfg.epochs):
        model.train()
        sums = {k: 0.0 for k in history}
        count = 0
        for batch in dl:
            x_window = batch["x_window"].to(device)
            x_now = batch["x_now"].to(device)
            x_next = batch["x_next"].to(device)

            out = model(x_window)
            recon = F.mse_loss(out["x_now_hat"], x_now)
            pred = F.mse_loss(out["x_next_hat"], x_next)
            sparse = out["adjacency"].mean()
            tgt = target_t.unsqueeze(0).expand(out["assign"].shape[0], -1, -1)
            align = alignment_loss(out["assign"], tgt)
            loss = (
                refine_cfg.lambda_recon * recon
                + refine_cfg.lambda_pred * pred
                + refine_cfg.lambda_align * align
                + refine_cfg.lambda_sparse * sparse
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if refine_cfg.grad_clip is not None and refine_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, refine_cfg.grad_clip)
            opt.step()

            bsz = len(x_window)
            for key, val in [("loss", loss), ("recon", recon), ("pred", pred), ("align", align), ("sparse", sparse)]:
                sums[key] += float(val.detach().cpu()) * bsz
            count += bsz

        for k in history:
            history[k].append(sums[k] / max(count, 1))
        print(
            f"refine {epoch + 1:03d}/{refine_cfg.epochs}: "
            f"loss={history['loss'][-1]:.4g} recon={history['recon'][-1]:.4g} "
            f"pred={history['pred'][-1]:.4g} align={history['align'][-1]:.4g}"
        )

    meta = {
        "mi_labels": labels,
        "cluster_to_slot": cluster_to_slot,
        "target": target_np,
        "history": history,
    }
    return model, meta


def choose_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    trace: np.ndarray,
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> Tuple[AgencyLatentModel, Dict[str, list]]:
    if train_cfg is None:
        train_cfg = TrainConfig()
    if model_cfg is None:
        model_cfg = ModelConfig(num_vars=trace.shape[1])
    if model_cfg.num_vars != trace.shape[1]:
        raise ValueError("model_cfg.num_vars must match trace.shape[1]")

    device = choose_device(train_cfg.device)
    ds = WindowTraceDataset(trace, model_cfg.window, model_cfg.horizon)
    dl = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)

    model = AgencyLatentModel(model_cfg).to(device)
    agency = AgencyRegularizer(model_cfg).to(device) if train_cfg.use_agency_regularizer else None
    params = list(model.parameters()) + ([] if agency is None else list(agency.parameters()))
    opt = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    history: Dict[str, list] = {"loss": [], "recon": [], "pred": [], "sparse": [], "control": [], "memory": [], "blanket": []}

    for epoch in range(train_cfg.epochs):
        model.train()
        if agency is not None:
            agency.train()

        sums = {k: 0.0 for k in history}
        count = 0

        for batch in dl:
            x_window = batch["x_window"].to(device)
            x_now = batch["x_now"].to(device)
            x_next = batch["x_next"].to(device)

            out = model(x_window)
            recon = F.mse_loss(out["x_now_hat"], x_now)
            pred = F.mse_loss(out["x_next_hat"], x_next)
            sparse = out["adjacency"].mean()
            loss = recon + pred + train_cfg.lambda_sparse * sparse

            control = x_now.new_tensor(0.0)
            memory = x_now.new_tensor(0.0)
            blanket = x_now.new_tensor(0.0)

            if agency is not None and epoch >= train_cfg.warmup_epochs:
                with torch.no_grad():
                    prev_window = x_window.clone()
                    if model_cfg.window >= 2:
                        prev_window[:, -1] = batch["x_prev"].to(device)
                    prev_slots, _ = model.encode(prev_window)

                reg = agency(out, prev_slots=prev_slots, epsilon=train_cfg.epsilon_blanket)
                control = reg["control"]
                memory = reg["memory"]
                blanket = reg["epsilon_blanket"]
                loss = (
                    loss
                    + train_cfg.lambda_control * control
                    + train_cfg.lambda_memory * memory
                    + train_cfg.lambda_epsilon_blanket * blanket
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
            opt.step()

            bsz = len(x_window)
            sums["loss"] += float(loss.detach().cpu()) * bsz
            sums["recon"] += float(recon.detach().cpu()) * bsz
            sums["pred"] += float(pred.detach().cpu()) * bsz
            sums["sparse"] += float(sparse.detach().cpu()) * bsz
            sums["control"] += float(control.detach().cpu()) * bsz
            sums["memory"] += float(memory.detach().cpu()) * bsz
            sums["blanket"] += float(blanket.detach().cpu()) * bsz
            count += bsz

        for k in history:
            history[k].append(sums[k] / max(count, 1))

        msg = " ".join(f"{k}={history[k][-1]:.4g}" for k in ["loss", "recon", "pred", "sparse", "control", "memory", "blanket"])
        print(f"epoch {epoch + 1:03d}/{train_cfg.epochs}: {msg}")

    return model, history


@torch.no_grad()
def encode_trace(
    model: AgencyLatentModel,
    trace: np.ndarray,
    batch_size: int = 256,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    cfg = model.cfg
    ds = WindowTraceDataset(trace, cfg.window, cfg.horizon, normalize=normalize)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

    slots, pred_slots, assign, adjacency = [], [], [], None
    for batch in dl:
        out = model(batch["x_window"].to(device))
        slots.append(out["slots"].cpu().numpy())
        pred_slots.append(out["pred_slots"].cpu().numpy())
        assign.append(out["assign"].cpu().numpy())
        adjacency = out["adjacency"].cpu().numpy()

    return {
        "slots": np.concatenate(slots, axis=0),          # [T-window, K, d]
        "pred_slots": np.concatenate(pred_slots, axis=0),
        "assign": np.concatenate(assign, axis=0),        # [T-window, K, N]
        "adjacency": adjacency,                          # [K, K]
        "mean": ds.mean,
        "std": ds.std,
    }


@dataclass
class TraceSimulationConfig:
    T: int = 5000
    num_agents: int = 3
    copies_per_role: int = 3
    decoy_vars: int = 8
    process_noise: float = 0.04
    observation_noise: float = 0.04
    redundancy_noise: float = 0.03
    interaction_strength: float = 0.45
    confound_strength: float = 0.25
    leakage_strength: float = 0.03
    mixing_strength: float = 0.05
    episodic: bool = True
    episode_len: int = 900
    episode_gap: int = 300
    seed: int = 0


@dataclass
class SimulationResult:
    trace: np.ndarray
    metadata: Dict[str, object]


def _ar1(T: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + sigma * rng.normal()
    return x


def _episode_mask(cfg: TraceSimulationConfig, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((cfg.T, cfg.num_agents), dtype=np.float32)
    if not cfg.episodic:
        return mask
    mask[:] = 0.0
    period = cfg.episode_len + cfg.episode_gap
    for k in range(cfg.num_agents):
        offset = int((k / max(cfg.num_agents, 1)) * period)
        for start in range(offset, cfg.T, period):
            end = min(start + cfg.episode_len, cfg.T)
            mask[start:end, k] = 1.0
    # Mild stochastic edge raggedness so episode boundaries are not perfectly clocklike.
    flips = rng.random(mask.shape) < 0.002
    mask[flips] = 1.0 - mask[flips]
    return mask


def simulate_known_agent_trace(cfg: TraceSimulationConfig = TraceSimulationConfig()) -> SimulationResult:
    """Generate simple raw traces with known UAD-friendly structure.

    The simulation intentionally contains only a few mechanisms:
    - each agent has one latent sensor, internal state, and action variable;
    - observed variables are noisy redundant copies of these latent roles;
    - actions from one agent drive sensors of another through a known directed ring;
    - a global AR(1) confounder drives many observed variables;
    - optional leakage creates epsilon-blankets rather than perfect blankets;
    - optional episodes make agentic structure appear and disappear.

    The returned trace has shape [T, N]. Metadata contains ground-truth role indices
    suitable for sanity-checking a UAD pipeline.
    """
    rng = np.random.default_rng(cfg.seed)
    T, K = cfg.T, cfg.num_agents

    global_confound = _ar1(T, rho=0.985, sigma=0.25, rng=rng)
    local_env = np.stack([_ar1(T, rho=0.96 - 0.02 * (k % 3), sigma=0.18, rng=rng) for k in range(K)], axis=1)
    active = _episode_mask(cfg, rng)

    direct_adjacency = np.zeros((K, K), dtype=np.float32)  # entry [target, source]
    for k in range(K):
        direct_adjacency[k, (k - 1) % K] = 1.0

    s = np.zeros((T, K), dtype=np.float32)
    h = np.zeros((T, K), dtype=np.float32)
    a = np.zeros((T, K), dtype=np.float32)

    policy_bias = rng.normal(0.0, 0.15, size=K).astype(np.float32)

    for t in range(1, T):
        incoming = a[t - 1] @ direct_adjacency.T
        for k in range(K):
            on = active[t, k]
            s[t, k] = on * (
                local_env[t, k]
                + cfg.interaction_strength * incoming[k]
                + cfg.confound_strength * global_confound[t]
                + cfg.process_noise * rng.normal()
            )
            h_drive = 0.82 * h[t - 1, k] + 0.55 * s[t, k] + 0.20 * a[t - 1, k]
            h[t, k] = on * np.tanh(h_drive + cfg.process_noise * rng.normal()) + (1.0 - on) * 0.92 * h[t - 1, k]
            a[t, k] = on * np.tanh(1.25 * h[t, k] - 0.25 * s[t, k] + policy_bias[k] + cfg.process_noise * rng.normal())

    columns = []
    var_names = []
    var_agent = []
    var_role = []
    role_indices: Dict[Tuple[int, str], list] = {}

    def add_var(name: str, agent: int, role: str, values: np.ndarray) -> None:
        idx = len(columns)
        columns.append(values.astype(np.float32))
        var_names.append(name)
        var_agent.append(agent)
        var_role.append(role)
        role_indices.setdefault((agent, role), []).append(idx)

    for k in range(K):
        other_action_mean = (a.sum(axis=1) - a[:, k]) / max(K - 1, 1)
        neighbor = (k + 1) % K
        bases = {
            "sensor": s[:, k],
            "internal": h[:, k] + cfg.leakage_strength * other_action_mean,
            "action": a[:, k],
        }
        for role, base in bases.items():
            for r in range(cfg.copies_per_role):
                coef = rng.normal(1.0, 0.08)
                mixed = cfg.mixing_strength * (s[:, neighbor] if role != "action" else a[:, neighbor])
                conf = cfg.confound_strength * (0.35 if role in {"sensor", "action"} else 0.15) * global_confound
                noise = (cfg.observation_noise + cfg.redundancy_noise * r) * rng.normal(size=T)
                add_var(f"agent{k}.{role}{r}", k, role, coef * base + mixed + conf + noise)

    for j in range(cfg.decoy_vars):
        if j % 3 == 0:
            values = global_confound + cfg.observation_noise * rng.normal(size=T)
            role = "global_confound_decoy"
        elif j % 3 == 1:
            values = _ar1(T, rho=0.93, sigma=0.30, rng=rng) + 0.2 * global_confound
            role = "autocorrelated_decoy"
        else:
            values = rng.normal(0.0, 1.0, size=T)
            role = "noise_decoy"
        add_var(f"decoy{j}.{role}", -1, role, values)

    trace = np.stack(columns, axis=1).astype(np.float32)
    trace = (trace - trace.mean(axis=0, keepdims=True)) / (trace.std(axis=0, keepdims=True) + 1e-6)

    agent_clusters = {
        k: sorted(role_indices[(k, "sensor")] + role_indices[(k, "internal")] + role_indices[(k, "action")])
        for k in range(K)
    }

    metadata: Dict[str, object] = {
        "var_names": var_names,
        "var_agent": np.array(var_agent, dtype=np.int64),
        "var_role": np.array(var_role, dtype=object),
        "role_indices": role_indices,
        "agent_clusters": agent_clusters,
        "direct_adjacency_target_source": direct_adjacency,
        "active_episode_mask": active,
        "latent_sensor": s,
        "latent_internal": h,
        "latent_action": a,
        "global_confound": global_confound,
        "config": cfg,
    }
    return SimulationResult(trace=trace, metadata=metadata)


def _cov(x: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    c = (x.T @ x) / max(len(x) - 1, 1)
    return c + ridge * np.eye(c.shape[0])


def _logdet_spd(c: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(c)
    if sign <= 0:
        c = c + 1e-4 * np.eye(c.shape[0])
        sign, val = np.linalg.slogdet(c)
    return float(val)


def _conditional_cov(x: np.ndarray, z: Optional[np.ndarray], ridge: float = 1e-5) -> np.ndarray:
    if z is None or z.shape[1] == 0:
        return _cov(x, ridge)
    xz = np.concatenate([x, z], axis=1)
    c = _cov(xz, ridge)
    dx = x.shape[1]
    c_xx = c[:dx, :dx]
    c_xz = c[:dx, dx:]
    c_zz = c[dx:, dx:]
    return c_xx - c_xz @ np.linalg.solve(c_zz, c_xz.T) + ridge * np.eye(dx)


def gaussian_cmi(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None, ridge: float = 1e-5) -> float:
    """Gaussian plug-in estimate of I(X;Y|Z), useful as a cheap UAD sanity check."""
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if z is not None:
        z = np.atleast_2d(z)
        if z.shape[0] != x.shape[0]:
            raise ValueError("z must have the same number of rows as x and y")
    sx = _conditional_cov(x, z, ridge)
    sy = _conditional_cov(y, z, ridge)
    sxy = _conditional_cov(np.concatenate([x, y], axis=1), z, ridge)
    return max(0.0, 0.5 * (_logdet_spd(sx) + _logdet_spd(sy) - _logdet_spd(sxy)))


def oracle_uad_scores(trace: np.ndarray, metadata: Dict[str, object], max_external_dim: int = 32) -> Dict[int, Dict[str, float]]:
    """Score known simulated agents using the UAD epsilon-blanket criterion.

    This is not the discovery algorithm. It is a compact regression test: if the
    trace is well generated, ground-truth agent clusters should have lower
    conditional leakage than raw cross-boundary dependence.
    """
    role_indices = metadata["role_indices"]
    agent_clusters = metadata["agent_clusters"]
    scores: Dict[int, Dict[str, float]] = {}
    all_idx = np.arange(trace.shape[1])

    for agent, cluster in agent_clusters.items():
        internal_idx = role_indices[(agent, "internal")]
        sensor_idx = role_indices[(agent, "sensor")]
        action_idx = role_indices[(agent, "action")]
        external_idx = np.setdiff1d(all_idx, np.array(cluster, dtype=np.int64))
        if len(external_idx) > max_external_dim:
            external_idx = external_idx[:max_external_dim]

        I_next = trace[1:, internal_idx]
        E_next = trace[1:, external_idx]
        SA_t = trace[:-1, sensor_idx + action_idx]

        raw_mi = gaussian_cmi(I_next, E_next, None)
        leakage = gaussian_cmi(I_next, E_next, SA_t)
        sep_ratio = leakage / (raw_mi + 1e-9)
        scores[int(agent)] = {
            "raw_mi": raw_mi,
            "conditional_leakage": leakage,
            "separation_ratio": sep_ratio,
            "num_cluster_vars": float(len(cluster)),
            "num_external_vars_used": float(len(external_idx)),
        }
    return scores


def lagged_action_sensor_matrix(trace: np.ndarray, metadata: Dict[str, object]) -> np.ndarray:
    """Ground-truth-role diagnostic: correlation from each agent's action copies to each agent's next sensors."""
    role_indices = metadata["role_indices"]
    K = int(metadata["config"].num_agents)
    out = np.zeros((K, K), dtype=np.float32)  # [target_sensor_agent, source_action_agent]
    for target in range(K):
        sidx = role_indices[(target, "sensor")]
        y = trace[1:, sidx].mean(axis=1)
        for source in range(K):
            aidx = role_indices[(source, "action")]
            x = trace[:-1, aidx].mean(axis=1)
            out[target, source] = np.corrcoef(x, y)[0, 1]
    return out


def score_against_ground_truth(
    clusters: Optional[Sequence[Dict[str, Any]]],
    metadata: Dict[str, object],
    assign: np.ndarray,
    learned_adjacency: Optional[np.ndarray] = None,
    assignment_threshold: float = 0.2,
) -> Dict[str, object]:
    """Compute simple latent-to-ground-truth recovery diagnostics.

    This function is intentionally lightweight: it does not assume a final
    slot-UAD implementation. It evaluates whether learned slots align with known
    simulated agents and whether learned slot-level adjacency is consistent with
    the simulator's directed inter-agent graph.
    """
    if assign.ndim != 3:
        raise ValueError("assign must have shape [T, K, N]")
    if not (0.0 <= assignment_threshold <= 1.0):
        raise ValueError("assignment_threshold must be in [0, 1]")

    avg_assign = assign.mean(axis=0)  # [K, N]
    num_slots, num_vars = avg_assign.shape
    var_agent = np.asarray(metadata["var_agent"], dtype=np.int64)
    if var_agent.shape[0] != num_vars:
        raise ValueError("metadata['var_agent'] size must match assign.shape[-1]")

    num_agents = int(metadata["config"].num_agents)
    non_decoy_mask = var_agent >= 0
    eps = 1e-9

    slot_agent_mass = np.zeros((num_slots, num_agents), dtype=np.float64)
    for a in range(num_agents):
        idx = np.where(var_agent == a)[0]
        if len(idx) > 0:
            slot_agent_mass[:, a] = avg_assign[:, idx].sum(axis=1)

    slot_total_mass = slot_agent_mass.sum(axis=1) + eps
    slot_best_agent = slot_agent_mass.argmax(axis=1)
    slot_purity = slot_agent_mass.max(axis=1) / slot_total_mass
    weighted_slot_purity = float((slot_purity * slot_total_mass).sum() / slot_total_mass.sum())
    active_slots = int((slot_total_mass > 1e-3).sum())

    # Agent concentration: how concentrated each true agent's variables are in one slot.
    agent_best_slot_mass = slot_agent_mass.max(axis=0)
    agent_total_mass = slot_agent_mass.sum(axis=0) + eps
    agent_concentration = agent_best_slot_mass / agent_total_mass

    # Soft variable-level classification through p(agent|slot) and assignments.
    p_agent_given_slot = slot_agent_mass / slot_total_mass[:, None]
    var_agent_scores = p_agent_given_slot.T @ avg_assign  # [A, N]
    predicted_agent_for_var = var_agent_scores.argmax(axis=0)
    var_acc_mask = non_decoy_mask
    var_agent_accuracy = float((predicted_agent_for_var[var_acc_mask] == var_agent[var_acc_mask]).mean())

    # Best slot-agent one-to-one matching score (greedy approximation).
    remaining_slot = np.ones(num_slots, dtype=bool)
    remaining_agent = np.ones(num_agents, dtype=bool)
    matched_mass = 0.0
    for _ in range(min(num_slots, num_agents)):
        masked = slot_agent_mass.copy()
        masked[~remaining_slot, :] = -1.0
        masked[:, ~remaining_agent] = -1.0
        k, a = np.unravel_index(masked.argmax(), masked.shape)
        if masked[k, a] < 0:
            break
        matched_mass += float(masked[k, a])
        remaining_slot[k] = False
        remaining_agent[a] = False
    matching_accuracy = float(matched_mass / (slot_agent_mass.sum() + eps))

    metrics: Dict[str, object] = {
        "slot_purity_mean": float(slot_purity.mean()),
        "slot_purity_weighted": weighted_slot_purity,
        "active_slots": float(active_slots),
        "agent_concentration_mean": float(agent_concentration.mean()),
        "agent_concentration_min": float(agent_concentration.min()),
        "variable_agent_accuracy": var_agent_accuracy,
        "slot_agent_matching_accuracy": matching_accuracy,
        "slot_best_agent": slot_best_agent.astype(int).tolist(),
    }

    if learned_adjacency is not None:
        learned_adjacency = np.asarray(learned_adjacency, dtype=np.float64)
        if learned_adjacency.shape != (num_slots, num_slots):
            raise ValueError("learned_adjacency must have shape [K, K]")
        true_adj = np.asarray(metadata["direct_adjacency_target_source"], dtype=np.float64)
        if true_adj.shape != (num_agents, num_agents):
            raise ValueError("metadata['direct_adjacency_target_source'] must have shape [A, A]")

        # Aggregate slot adjacency to agent adjacency using soft slot-agent memberships.
        agent_adj = np.zeros((num_agents, num_agents), dtype=np.float64)
        for target in range(num_agents):
            for source in range(num_agents):
                wt = p_agent_given_slot[:, target]  # target slots
                ws = p_agent_given_slot[:, source]  # source slots
                denom = float(np.outer(wt, ws).sum()) + eps
                agent_adj[target, source] = float((learned_adjacency * np.outer(wt, ws)).sum() / denom)

        offdiag = ~np.eye(num_agents, dtype=bool)
        true_vals = true_adj[offdiag]
        pred_vals = agent_adj[offdiag]
        true_edge_mask = true_vals > 0.5
        false_edge_mask = ~true_edge_mask
        mean_true = float(pred_vals[true_edge_mask].mean()) if true_edge_mask.any() else 0.0
        mean_false = float(pred_vals[false_edge_mask].mean()) if false_edge_mask.any() else 0.0
        edge_corr = float(np.corrcoef(true_vals, pred_vals)[0, 1]) if pred_vals.std() > 1e-12 else 0.0
        metrics["edge_recovery"] = {
            "mean_pred_on_true_edges": mean_true,
            "mean_pred_on_false_edges": mean_false,
            "edge_separation": mean_true - mean_false,
            "edge_correlation": edge_corr,
            "agent_level_adjacency": agent_adj.tolist(),
        }

    if clusters is not None and len(clusters) > 0:
        true_clusters = metadata["agent_clusters"]
        cluster_scores = []
        for c in clusters:
            slot_idx = c.get("slots", [])
            if len(slot_idx) == 0:
                continue
            slot_idx = np.array(slot_idx, dtype=np.int64)
            if (slot_idx < 0).any() or (slot_idx >= num_slots).any():
                continue
            var_score = avg_assign[slot_idx].sum(axis=0) / max(len(slot_idx), 1)
            pred_var_set = set(np.where(var_score >= assignment_threshold)[0].tolist())
            best_agent = -1
            best_jaccard = 0.0
            for agent, idxs in true_clusters.items():
                true_set = set(int(i) for i in idxs)
                inter = len(pred_var_set & true_set)
                union = len(pred_var_set | true_set)
                jaccard = (inter / union) if union > 0 else 0.0
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_agent = int(agent)
            cluster_scores.append(
                {
                    "slots": slot_idx.tolist(),
                    "best_agent": best_agent,
                    "best_jaccard": float(best_jaccard),
                    "num_pred_vars": float(len(pred_var_set)),
                }
            )
        metrics["cluster_alignment"] = {
            "num_clusters_scored": float(len(cluster_scores)),
            "mean_best_jaccard": float(np.mean([c["best_jaccard"] for c in cluster_scores])) if cluster_scores else 0.0,
            "details": cluster_scores,
        }

    return metrics


def synthetic_trace(T: int = 4000, N: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((T, N), dtype=np.float32)
    groups = [range(0, 8), range(8, 16), range(16, 24)]
    phases = rng.normal(size=(3,)).astype(np.float32)

    for t in range(1, T):
        drivers = np.array([
            math.sin(0.025 * t + phases[0]),
            math.sin(0.017 * t + phases[1]),
            math.sin(0.011 * t + phases[2]),
        ], dtype=np.float32)
        drivers[1] += 0.3 * drivers[0] if t % 200 < 120 else 0.0
        drivers[2] += 0.2 * drivers[1] if t % 350 < 80 else 0.0
        for gi, idxs in enumerate(groups):
            for j in idxs:
                x[t, j] = 0.85 * x[t - 1, j] + 0.15 * drivers[gi] + 0.03 * rng.normal()
    return x


if __name__ == "__main__":
    sim = simulate_known_agent_trace(
        TraceSimulationConfig(
            T=5000,
            num_agents=3,
            copies_per_role=3,
            decoy_vars=8,
            interaction_strength=0.45,
            confound_strength=0.25,
            leakage_strength=0.03,
            mixing_strength=0.05,
            episodic=True,
            seed=1,
        )
    )
    trace = sim.trace
    print("trace", trace.shape)
    print("oracle UAD scores", oracle_uad_scores(trace, sim.metadata))
    print("lagged action->sensor corr", lagged_action_sensor_matrix(trace, sim.metadata))

    # Fewer slots reduce splitting one true agent across many latent slots.
    model_cfg = ModelConfig(num_vars=trace.shape[1], window=16, num_slots=6, slot_dim=16)
    train_cfg = TrainConfig(
        # Extra epochs help assignment specialization stabilize.
        epochs=35,
        batch_size=128,
        use_agency_regularizer=False,
        device=None,
    )
    model, history = train_model(trace, model_cfg, train_cfg)
    latent = encode_trace(model, trace)
    print("slots", latent["slots"].shape)
    print("assign", latent["assign"].shape)
    print("adjacency", latent["adjacency"].shape)
    gt_scores = score_against_ground_truth(
        clusters=None,
        metadata=sim.metadata,
        assign=latent["assign"],
        learned_adjacency=latent["adjacency"],
    )
    print("chance variable-agent accuracy", 1.0 / sim.metadata["config"].num_agents)
    print("ground-truth alignment", gt_scores)
