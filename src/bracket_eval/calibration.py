"""Fit the policy-table confidence thresholds on held-out data.

WHY THIS MODULE EXISTS
----------------------
``src/policy_decision/table.py`` buckets confidence at 0.4 and 0.7:

    _LOW_THRESHOLD  = 0.4
    _HIGH_THRESHOLD = 0.7

Those two numbers select which row of a 12-cell table fires, and one of the
cells they can select is ``blocked``. There is no derivation for either value
anywhere in the repository. They are round numbers that look reasonable.

This module replaces the guess with a fit. It sweeps candidate threshold pairs,
scores each against a safety-weighted cost function on held-out conversations,
and reports the pair that minimises expected harm.

WHY NOT JUST MAXIMISE ACCURACY
------------------------------
Because the errors are not symmetric and accuracy pretends they are.

    Calling a child an adult      -> a minor gets an unrestricted product.
    Calling an adult a child      -> an adult sees a tightened tone and is asked
                                     to confirm their age.

The first is a child-safety failure. The second is a mildly annoying prompt with
a one-tap resolution -- and the architecture already has a step-up ladder built
precisely to absorb it. Treating those as equal-cost errors, which is exactly
what argmax-accuracy does, optimises for the wrong thing.

So the cost function weights them asymmetrically, and the weights are explicit
module constants rather than being buried in a formula -- a jurisdiction with
different regulatory exposure should be able to retune them by editing two
numbers and re-running, without touching the search.

WHAT THIS IS NOT
----------------
This is not temperature scaling or isotonic regression. Those reshape the
confidence score itself, which would break the "confidence is a transparent
arithmetic function of evidence" property that makes the system auditable.
Fitting the THRESHOLDS instead leaves the confidence function untouched and
keeps the whole path inspectable: a fitted threshold is still a constant in a
table, still deterministic, still explainable to a regulator.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bracket_eval.pipeline import ConversationResult

# ---------------------------------------------------------------------------
# Asymmetric error costs
#
# Tunable per product and jurisdiction. Relative magnitude is what matters, not
# the absolute scale. Defaults encode: letting a minor through is roughly an
# order of magnitude worse than over-restricting an adult.
# ---------------------------------------------------------------------------

COST_CHILD_TREATED_AS_ADULT: float = 10.0
COST_TEEN_TREATED_AS_ADULT: float = 6.0
COST_ADULT_OVER_RESTRICTED: float = 1.0
COST_MINOR_UNDER_PROTECTED: float = 3.0  # minor detected, but posture too loose
COST_ABSTENTION_ON_MINOR: float = 2.0  # abstained when the user was a minor

# Postures that leave a product effectively open.
_OPEN_POSTURES: frozenset[str] = frozenset({"standard"})
# Postures that meaningfully restrict.
_PROTECTIVE_POSTURES: frozenset[str] = frozenset({"restricted", "blocked"})

MINOR_BANDS: frozenset[str] = frozenset({"child", "teen"})


@dataclass(frozen=True)
class ThresholdCandidate:
    """One (low, high) threshold pair and its measured cost."""

    low_threshold: float
    high_threshold: float
    total_cost: float
    accuracy: float
    minors_treated_as_adult: int
    adults_over_restricted: int
    abstention_rate: float

    @property
    def mean_cost(self) -> float:
        return self.total_cost


def _posture_for(band: str, confidence: float, low: float, high: float) -> str:
    """Re-derive the posture under candidate thresholds.

    Mirrors ``policy_decision.table`` exactly, but with the two cut points
    injected rather than read from module constants -- that injection is the
    only reason this function exists instead of calling lookup() directly.

    Kept deliberately as a literal transcription of the real table so a drift
    between the two is visible on inspection rather than hidden in logic.
    """
    if confidence < low:
        bucket = "low"
    elif confidence < high:
        bucket = "medium"
    else:
        bucket = "high"

    table = {
        ("unknown", "low"): "standard",
        ("unknown", "medium"): "standard",
        ("unknown", "high"): "caution",
        ("adult", "low"): "standard",
        ("adult", "medium"): "standard",
        ("adult", "high"): "standard",
        ("teen", "low"): "caution",
        ("teen", "medium"): "restricted",
        ("teen", "high"): "restricted",
        ("child", "low"): "caution",
        ("child", "medium"): "restricted",
        ("child", "high"): "blocked",
    }
    return table.get((band, bucket), "standard")


def score_thresholds(
    results: list[ConversationResult],
    low_threshold: float,
    high_threshold: float,
) -> ThresholdCandidate:
    """Score one threshold pair against replayed conversations.

    Cost is accumulated per conversation from the FINAL turn state, since that
    is the posture the user ends the session under.
    """
    total_cost = 0.0
    correct_predictions = 0
    minors_treated_as_adult = 0
    adults_over_restricted = 0
    abstentions = 0

    for result in results:
        true_band = result.true_band
        predicted_band = result.predicted_band
        confidence = result.final_confidence
        posture = _posture_for(predicted_band, confidence, low_threshold, high_threshold)

        if true_band == predicted_band:
            correct_predictions += 1

        if predicted_band == "unknown":
            abstentions += 1

        # --- unsafe errors: a minor ends the session in an open posture ---
        if true_band in MINOR_BANDS and posture in _OPEN_POSTURES:
            if predicted_band == "adult":
                minors_treated_as_adult += 1
                total_cost += (
                    COST_CHILD_TREATED_AS_ADULT
                    if true_band == "child"
                    else COST_TEEN_TREATED_AS_ADULT
                )
            elif predicted_band == "unknown":
                # Abstention is a safe failure but still a failure -- the
                # protective signal the product needed was never produced.
                total_cost += COST_ABSTENTION_ON_MINOR
            else:
                # Band was right but confidence was too low to act on it.
                total_cost += COST_MINOR_UNDER_PROTECTED

        # --- annoying errors: an adult gets meaningfully restricted ---
        elif true_band == "adult" and posture in _PROTECTIVE_POSTURES:
            adults_over_restricted += 1
            total_cost += COST_ADULT_OVER_RESTRICTED

    sample_count = len(results) or 1
    return ThresholdCandidate(
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        total_cost=total_cost / sample_count,
        accuracy=correct_predictions / sample_count,
        minors_treated_as_adult=minors_treated_as_adult,
        adults_over_restricted=adults_over_restricted,
        abstention_rate=abstentions / sample_count,
    )


def fit_thresholds(
    results: list[ConversationResult],
    step: float = 0.05,
    minimum_separation: float = 0.10,
) -> tuple[ThresholdCandidate, list[ThresholdCandidate]]:
    """Grid-search the (low, high) threshold pair that minimises expected cost.

    A grid search rather than gradient descent because the cost surface is a
    step function -- it only changes when a threshold crosses an actual
    confidence value in the data, so it has zero gradient almost everywhere.
    The search space is small enough (a few hundred pairs) that exhaustive is
    both simpler and exact.

    ``minimum_separation`` stops the search collapsing the medium bucket to
    nothing. A degenerate low==high would technically be scoreable but would
    silently delete the entire "caution" tier from the policy table.

    Returns the best candidate and the full sorted list, because the runner-up
    pairs are informative: a broad flat optimum means the thresholds do not
    matter much, while a sharp one means they are load-bearing and need real
    data behind them.
    """
    candidates: list[ThresholdCandidate] = []

    steps_in_range = int(1.0 / step)
    for low_step in range(1, steps_in_range):
        low_threshold = round(low_step * step, 4)
        for high_step in range(low_step + 1, steps_in_range):
            high_threshold = round(high_step * step, 4)
            if high_threshold - low_threshold < minimum_separation:
                continue
            candidates.append(
                score_thresholds(results, low_threshold, high_threshold)
            )

    if not candidates:
        raise ValueError("no valid threshold pairs in search space")

    # Sort by cost, then prefer higher accuracy, then prefer the lower
    # low-threshold as a deterministic tie-break so repeated runs on the same
    # data always return the same pair.
    candidates.sort(key=lambda c: (c.total_cost, -c.accuracy, c.low_threshold))
    return candidates[0], candidates


def render_comparison(
    fitted: ThresholdCandidate, baseline: ThresholdCandidate
) -> str:
    """Side-by-side of fitted thresholds against the upstream hardcoded pair."""
    lines = [
        "                          upstream      fitted     change",
        "  " + "-" * 52,
        f"  low threshold        {baseline.low_threshold:11.2f}{fitted.low_threshold:12.2f}",
        f"  high threshold       {baseline.high_threshold:11.2f}{fitted.high_threshold:12.2f}",
        "",
        f"  weighted cost        {baseline.total_cost:11.3f}{fitted.total_cost:12.3f}"
        f"{fitted.total_cost - baseline.total_cost:+11.3f}",
        f"  band accuracy        {baseline.accuracy:11.3f}{fitted.accuracy:12.3f}"
        f"{fitted.accuracy - baseline.accuracy:+11.3f}",
        f"  minors as adult      {baseline.minors_treated_as_adult:11d}"
        f"{fitted.minors_treated_as_adult:12d}"
        f"{fitted.minors_treated_as_adult - baseline.minors_treated_as_adult:+11d}",
        f"  adults restricted    {baseline.adults_over_restricted:11d}"
        f"{fitted.adults_over_restricted:12d}"
        f"{fitted.adults_over_restricted - baseline.adults_over_restricted:+11d}",
    ]
    return "\n".join(lines)
