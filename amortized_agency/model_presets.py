"""Named model scales for learned-method sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ContextScale:
    enc_dim: int = 64
    conv_ch: int = 32
    n_tokens: int = 16
    n_heads: int = 4
    n_layers: int = 2


@dataclass(frozen=True)
class SiameseScale:
    embed_dim: int = 64
    hidden: int = 64


CONTEXT_SCALES: Dict[str, ContextScale] = {
    "base": ContextScale(),
    "large": ContextScale(enc_dim=128, conv_ch=64, n_tokens=24, n_heads=8, n_layers=3),
    "xl": ContextScale(enc_dim=256, conv_ch=128, n_tokens=32, n_heads=8, n_layers=4),
}

SIAMESE_SCALES: Dict[str, SiameseScale] = {
    "base": SiameseScale(),
    "large": SiameseScale(embed_dim=128, hidden=128),
    "xl": SiameseScale(embed_dim=256, hidden=256),
}


def build_context_model(window: int, scale: str = "base"):
    from amortized_agency.context_model import ContextualAffinityModel

    s = CONTEXT_SCALES[scale]
    return ContextualAffinityModel(
        window,
        enc_dim=s.enc_dim,
        conv_ch=s.conv_ch,
        n_tokens=s.n_tokens,
        n_heads=s.n_heads,
        n_layers=s.n_layers,
    )


def build_siamese_model(window: int, scale: str = "base"):
    from amortized_agency.siamese import SiameseAffinityModel

    s = SIAMESE_SCALES[scale]
    return SiameseAffinityModel(window, embed_dim=s.embed_dim, hidden=s.hidden)
