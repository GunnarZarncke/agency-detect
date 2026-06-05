"""uad_worm — Unsupervised Agent Discovery on C. elegans whole-brain data (E20).

See README.md for the scoped plan. M0 (this commit): core information-theoretic
estimator + blanket loss + nulls + synthetic benchmarks, validated offline before any
worm data is touched.
"""

from uad_worm.cmi import gaussian_cmi
from uad_worm.blanket import blanket_loss, random_partition_null, blanket_pvalue

__all__ = [
    "gaussian_cmi",
    "blanket_loss",
    "random_partition_null",
    "blanket_pvalue",
]
