"""value_detect — unsupervised value discovery (directional UAD).

Chunk 2 surface: wrap Gunnar's read-only handle-world, record passive traces, and
produce plain-English audit artifacts (story printout, change-frequency chart).
"""

from __future__ import annotations

from .changes import (
    change_frequencies,
    mechanism_agreement,
    plot_change_frequencies,
    text_bar_chart,
)
from .directed import (
    binary_entropy,
    cmi,
    conservation_check,
    directed_information,
    mi,
    transfer_entropy,
)
from .agentblocks import BlockScorer, compress_block, detect_blocks, detect_blocks_swept
from .criteria import (
    evaluate_goalprogress_test,
    evaluate_main_test,
    evaluate_nocore_test,
    evaluate_scramble_test,
    signature_flags,
)
from .floors import ConventionScorer, null_floors, shift_null_floors
from .narrate import goal_flip_report, narrate_window
from .scorer import (
    drop_aliases,
    push_in_megastate,
    score_trace,
    score_trace_fused,
    score_trace_fused_bestkey,
)
from .variants import (
    NoCoreHandleWorld,
    calibration_frame,
    passive_trace_nocore,
    scramble_frame,
)
from .yardstick import operated_colony_rollout, yardstick_scores, yardstick_verdict
from .worlds_v2 import (
    alias_colony_frame,
    colony_frame,
    deep_synergy_frame,
    slow_meter_frame,
)
from .world import (
    BURN_IN,
    DEFAULT_NOISE,
    Trace,
    passive_trace,
    record_trace,
    variable_names,
    verify_world_defaults,
)

__all__ = [
    "BURN_IN",
    "DEFAULT_NOISE",
    "Trace",
    "passive_trace",
    "record_trace",
    "variable_names",
    "verify_world_defaults",
    "change_frequencies",
    "mechanism_agreement",
    "plot_change_frequencies",
    "text_bar_chart",
    "narrate_window",
    "goal_flip_report",
    "cmi",
    "mi",
    "transfer_entropy",
    "directed_information",
    "conservation_check",
    "binary_entropy",
    "score_trace",
    "score_trace_fused",
    "score_trace_fused_bestkey",
    "push_in_megastate",
    "drop_aliases",
    "ConventionScorer",
    "null_floors",
    "shift_null_floors",
    "signature_flags",
    "evaluate_main_test",
    "evaluate_nocore_test",
    "evaluate_scramble_test",
    "evaluate_goalprogress_test",
    "NoCoreHandleWorld",
    "passive_trace_nocore",
    "scramble_frame",
    "calibration_frame",
    "BlockScorer",
    "detect_blocks",
    "detect_blocks_swept",
    "compress_block",
    "colony_frame",
    "alias_colony_frame",
    "deep_synergy_frame",
    "slow_meter_frame",
    "operated_colony_rollout",
    "yardstick_scores",
    "yardstick_verdict",
]
