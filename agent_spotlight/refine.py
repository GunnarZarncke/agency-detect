"""Align a trained slot model to a single MI cluster (spotlight refine)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learn_agents.learn_agents import (
    AgencyLatentModel,
    RefineConfig,
    WindowTraceDataset,
    _mean_assignment,
    alignment_loss,
    build_mi_alignment_target,
    choose_device,
    encode_trace,
    match_mi_clusters_to_slots,
)

from .config import SpotlightConfig


def build_single_cluster_labels(n_vars: int, cluster_var_indices: List[int]) -> np.ndarray:
    labels = np.full(n_vars, -1, dtype=np.int64)
    for j in cluster_var_indices:
        labels[int(j)] = 0
    return labels


def refine_to_cluster(
    model: AgencyLatentModel,
    trace: np.ndarray,
    cluster_var_indices: List[int],
    cfg: SpotlightConfig,
) -> tuple[AgencyLatentModel, Dict[str, Any]]:
    """Refine so spotlight slot(s) align to one MI cluster only."""
    labels = build_single_cluster_labels(trace.shape[1], cluster_var_indices)
    refine_cfg = RefineConfig(
        epochs=cfg.refine_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.refine_lr,
        lambda_align=cfg.lambda_align,
        lambda_sparse=cfg.lambda_sparse,
        device=cfg.device,
        mi_fixed_k=1,
        mi_background_factorize=False,
        target_smoothing=cfg.target_smoothing,
    )

    device = choose_device(cfg.device)
    model = model.to(device)
    avg_assign = _mean_assignment(model, trace, batch_size=cfg.batch_size)
    cluster_to_slot = match_mi_clusters_to_slots(labels, avg_assign)
    if 0 not in cluster_to_slot:
        cluster_to_slot[0] = int(cfg.spotlight_slot_index)

    target_np = build_mi_alignment_target(
        labels, model.cfg.num_slots, cluster_to_slot, smoothing=cfg.target_smoothing
    )
    target_t = torch.from_numpy(target_np).to(device)

    for p in model.var_encoder.parameters():
        p.requires_grad = False
    for p in model.dynamics.parameters():
        p.requires_grad = False
    model.dynamics.edge_logits.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=refine_cfg.lr, weight_decay=refine_cfg.weight_decay)
    ds = WindowTraceDataset(trace, model.cfg.window, model.cfg.horizon)
    dl = DataLoader(ds, batch_size=refine_cfg.batch_size, shuffle=True, drop_last=True)

    history: Dict[str, list] = {"loss": [], "recon": [], "pred": [], "align": [], "sparse": []}

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
            if refine_cfg.grad_clip and refine_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, refine_cfg.grad_clip)
            opt.step()
            bsz = len(x_window)
            for key, val in [
                ("loss", loss),
                ("recon", recon),
                ("pred", pred),
                ("align", align),
                ("sparse", sparse),
            ]:
                sums[key] += float(val.detach().cpu()) * bsz
            count += bsz
        for k in history:
            history[k].append(sums[k] / max(count, 1))

    meta = {
        "cluster_to_slot": cluster_to_slot,
        "cluster_var_indices": cluster_var_indices,
        "history": history,
        "slot_assignment_mean": encode_trace(model, trace)["assign"].mean(axis=0).tolist(),
    }
    return model, meta
