"""Shared validation helpers for Bracket contracts.

These validators enforce invariants that cannot be expressed purely in Pydantic:
- BandEstimate must never carry a confidence value (LLM must not produce one).
- PlannerAction must have a known action_type (fail-closed on unknown actions).
"""

from __future__ import annotations

from src.contracts.models import BandEstimate, PlannerAction

_FORBIDDEN_ESTIMATE_KEYS = frozenset({"confidence", "confidence_score", "conf"})

_VALID_ACTION_TYPES = frozenset(
    {
        "gate_check",
        "read_evidence",
        "update_evidence",
        "compute_confidence",
        "policy_decide",
        "emit_posture",
        "persist_confirmed",
        "delegate_extract",
        "delegate_estimate",
        "delegate_stepup",
        "finish",
    }
)


def validate_band_estimate(raw: dict[str, object]) -> BandEstimate:
    """Parse and validate an BandEstimate from a raw dict.

    Raises ValueError if any forbidden key (confidence, confidence_score, conf)
    is present — this enforces the invariant that the LLM must NEVER emit a
    confidence value. Confidence is always computed deterministically in Python.
    """
    forbidden = _FORBIDDEN_ESTIMATE_KEYS & raw.keys()
    if forbidden:
        raise ValueError(
            f"BandEstimate must not contain confidence key(s): {forbidden}. "
            "Confidence is computed deterministically in confidence.py — "
            "the LLM must not produce it."
        )
    return BandEstimate.model_validate(raw)


def validate_planner_action(raw: dict[str, object]) -> PlannerAction:
    """Parse and validate a PlannerAction from a raw dict.

    Raises ValueError for unknown action_type values (fail closed).
    Also rejects extra fields via the model's ConfigDict(extra='forbid').
    """
    action_type = raw.get("action_type")
    if action_type not in _VALID_ACTION_TYPES:
        raise ValueError(
            f"Unknown PlannerAction.action_type: {action_type!r}. "
            f"Valid types: {sorted(_VALID_ACTION_TYPES)}"
        )
    return PlannerAction.model_validate(raw)
