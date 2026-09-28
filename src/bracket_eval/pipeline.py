"""Deterministic end-to-end pipeline driver for evaluation.

WHY THIS MODULE EXISTS
----------------------
The upstream eval script (``scripts/eval_pipeline_against_synthetic.py``) is
async, requires a live LLM endpoint, and is explicitly excluded from CI. That
makes it useless for the thing we actually need: replaying thousands of labelled
transcripts quickly, on a laptop, with no GPU and no network.

This driver walks the same production modules in the same order --

    signal_extraction -> evidence_fabric -> band_inference -> policy_decision

-- but takes the deterministic branch at every LLM boundary. Same lexicon, same
weights, same confidence formula, same policy table. Nothing is reimplemented or
approximated here; if a number comes out of this driver it came out of the real
scoring path.

Being synchronous and dependency-free is the point: the calibration harness runs
it tens of thousands of times in a sweep, and that has to take seconds.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from src.band_inference import rule_estimator
from src.band_inference.confidence import compute_confidence
from src.contracts.models import Decision
from src.evidence_fabric.service import EvidenceFabricService
from src.policy_decision.table import lookup as policy_lookup
from src.signal_extraction.keyword_extractor import extract_cues


@dataclass(frozen=True)
class TurnResult:
    """Pipeline state captured after a single turn.

    Retaining per-turn results (rather than only the final state) is what lets
    the calibration harness measure how confidence evolves across a
    conversation, and lets the evasion suite verify that a band never settles
    on "adult" at ANY point -- not merely at the end.
    """

    turn_index: int
    text: str
    band: str
    confidence: float
    decision: Decision
    cue_subtypes: tuple[str, ...]
    evasion_flag: bool
    evasion_patterns: tuple[str, ...]


@dataclass(frozen=True)
class ConversationResult:
    """Full replay of one conversation through the pipeline."""

    conversation_id: str
    true_band: str
    turns: tuple[TurnResult, ...]

    @property
    def final(self) -> TurnResult:
        """The last turn -- the pipeline's settled view of this conversation."""
        return self.turns[-1]

    @property
    def predicted_band(self) -> str:
        return self.final.band

    @property
    def final_confidence(self) -> float:
        return self.final.confidence

    @property
    def ever_predicted_adult(self) -> bool:
        """True when ANY turn settled on 'adult'.

        Used by the evasion suite: an adversarial child transcript that touches
        'adult' even once has already failed, because the host product would
        have opened up features on that turn.
        """
        return any(turn.band == "adult" for turn in self.turns)


def replay_conversation(
    turn_texts: list[str],
    true_band: str = "unknown",
    conversation_id: str | None = None,
) -> ConversationResult:
    """Replay *turn_texts* through the deterministic pipeline, turn by turn.

    Each conversation gets a fresh EvidenceFabricService so sessions cannot
    leak evidence into each other -- an isolation bug here would silently
    inflate accuracy on later conversations in a batch.
    """
    conversation_id = conversation_id or f"eval-{uuid.uuid4().hex[:12]}"

    # Fresh fabric per conversation: evidence is session-scoped by design and
    # the eval must honour that, not accumulate across the whole dataset.
    evidence_fabric = EvidenceFabricService()

    turn_results: list[TurnResult] = []

    for turn_index, text in enumerate(turn_texts):
        # M2 -- deterministic cue extraction (English or Hinglish lexicon).
        signal_set = extract_cues(text)

        # M3 -- accumulate into session evidence with decay and corroboration.
        evidence = evidence_fabric.update(conversation_id, signal_set)

        # M4a -- rule estimator proposes a band. It never emits confidence.
        estimate = rule_estimator.estimate(evidence)

        # M4b -- confidence computed in pure Python from evidence shape.
        # This is the invariant the whole architecture rests on, so the eval
        # deliberately routes through the real function rather than a stub.
        confidence = compute_confidence(evidence, estimate)

        # M5 -- deterministic policy table lookup.
        decision = policy_lookup(estimate.band, confidence)

        turn_results.append(
            TurnResult(
                turn_index=turn_index,
                text=text,
                band=estimate.band,
                confidence=confidence,
                decision=decision,
                cue_subtypes=tuple(
                    cue.subtype for cue in signal_set.cues if cue.subtype
                ),
                evasion_flag=estimate.evasion_flag,
                evasion_patterns=tuple(getattr(estimate, "evasion_patterns", ()) or ()),
            )
        )

    return ConversationResult(
        conversation_id=conversation_id,
        true_band=true_band,
        turns=tuple(turn_results),
    )


def replay_dataset(conversations: list[dict]) -> list[ConversationResult]:
    """Replay a list of ``{"id", "band", "turns"}`` dicts.

    Kept as a thin loop rather than parallelised: the deterministic path is
    already fast enough that process-pool overhead would dominate, and a single
    process keeps results trivially reproducible.
    """
    return [
        replay_conversation(
            turn_texts=conversation["turns"],
            true_band=conversation["band"],
            conversation_id=conversation.get("id"),
        )
        for conversation in conversations
    ]
