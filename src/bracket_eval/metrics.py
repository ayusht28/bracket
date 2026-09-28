"""Classification and calibration metrics.

WHY THIS MODULE EXISTS
----------------------
The upstream system's central safety claim is:

    "Confidence is computed deterministically in Python, never taken from the
     LLM, therefore the model cannot emit a confident-but-wrong verdict."

The first half is true and structurally enforced. The second half does not
follow from it. Determinism guarantees that the same input yields the same
number; it guarantees nothing about whether that number MEANS anything. A
deterministic function that returns 0.9 for everything is perfectly
deterministic and perfectly useless.

The property actually needed is CALIBRATION: among all turns scored 0.7, roughly
70% should be correct. Without that, the thresholds in ``policy_decision/table.py``
(0.4 and 0.7) are arbitrary cut points on an uninterpretable scale -- and one of
the cells they select is ``blocked``. Gating a block on an uncalibrated score is
precisely the failure mode the architecture claims to have designed out.

This module measures it. Pure standard library -- no numpy, no sklearn -- so it
runs anywhere the pipeline runs and adds no install burden.

METRICS IMPLEMENTED
-------------------
Expected Calibration Error (ECE)
    Weighted mean gap between confidence and accuracy across bins. The headline
    number. Lower is better; below ~0.05 is usually considered well calibrated.

Maximum Calibration Error (MCE)
    Worst single-bin gap. Matters more than ECE for safety work, because a
    single badly-miscalibrated bin is a systematic failure even when the
    average looks fine.

Brier score
    Mean squared error of the confidence estimate. Captures sharpness and
    calibration together, so it penalises a model that is well calibrated only
    because it always says 0.5.

Reliability bins
    The raw per-bin table the two errors are computed from. Kept in the output
    because the aggregate numbers hide direction -- over- vs under-confidence
    call for opposite fixes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# Bands that participate in accuracy scoring. "unknown" is excluded because it
# is an abstention, not a prediction -- see confusion_matrix() for how it is
# accounted separately.
SCORED_BANDS: tuple[str, ...] = ("child", "teen", "adult")

DEFAULT_BIN_COUNT: int = 10


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


@dataclass
class ConfusionMatrix:
    """Counts of (true band, predicted band) pairs, including abstentions."""

    counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    labels: tuple[str, ...] = SCORED_BANDS + ("unknown",)

    def add(self, true_band: str, predicted_band: str) -> None:
        self.counts[(true_band, predicted_band)] += 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def accuracy(self) -> float:
        """Fraction of predictions that matched the true band exactly.

        Abstentions ("unknown" predicted) count as WRONG here. That is the
        strict reading and the honest one -- an abstention is a failure to
        produce the protective signal, even though it is a safe failure.
        See ``accuracy_excluding_abstentions`` for the lenient view.
        """
        if self.total == 0:
            return 0.0
        correct = sum(
            count for (true, pred), count in self.counts.items() if true == pred
        )
        return correct / self.total

    @property
    def abstention_rate(self) -> float:
        """Fraction of conversations where the pipeline predicted 'unknown'."""
        if self.total == 0:
            return 0.0
        abstained = sum(
            count for (_, pred), count in self.counts.items() if pred == "unknown"
        )
        return abstained / self.total

    @property
    def accuracy_excluding_abstentions(self) -> float:
        """Accuracy over conversations where the pipeline actually committed."""
        committed = [
            (true, pred, count)
            for (true, pred), count in self.counts.items()
            if pred != "unknown"
        ]
        committed_total = sum(count for _, _, count in committed)
        if committed_total == 0:
            return 0.0
        correct = sum(count for true, pred, count in committed if true == pred)
        return correct / committed_total

    def per_band(self) -> dict[str, dict[str, float]]:
        """Precision / recall / F1 for each scored band."""
        results: dict[str, dict[str, float]] = {}
        for band in SCORED_BANDS:
            true_positives = self.counts.get((band, band), 0)
            predicted_as_band = sum(
                count for (_, pred), count in self.counts.items() if pred == band
            )
            actually_band = sum(
                count for (true, _), count in self.counts.items() if true == band
            )
            precision = true_positives / predicted_as_band if predicted_as_band else 0.0
            recall = true_positives / actually_band if actually_band else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            results[band] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": actually_band,
            }
        return results

    def child_missed_as_adult(self) -> int:
        """Count of the single worst error: a child predicted as an adult.

        Surfaced as its own metric because aggregate accuracy treats this the
        same as calling an adult a teen, and they are not remotely the same.
        This is the number that should gate a release.
        """
        return self.counts.get(("child", "adult"), 0)

    def minor_missed_as_adult(self) -> int:
        """Children OR teens predicted as adult -- the full unsafe-error count."""
        return self.counts.get(("child", "adult"), 0) + self.counts.get(
            ("teen", "adult"), 0
        )

    def render(self) -> str:
        """Human-readable matrix. Rows are true bands, columns predicted."""
        header = "true \\ pred".ljust(14) + "".join(
            label.rjust(10) for label in self.labels
        )
        lines = [header, "-" * len(header)]
        for true_band in self.labels:
            row = true_band.ljust(14)
            for predicted_band in self.labels:
                row += str(self.counts.get((true_band, predicted_band), 0)).rjust(10)
            lines.append(row)
        return "\n".join(lines)


def confusion_matrix(
    true_bands: list[str], predicted_bands: list[str]
) -> ConfusionMatrix:
    """Build a ConfusionMatrix from parallel label lists."""
    if len(true_bands) != len(predicted_bands):
        raise ValueError(
            f"length mismatch: {len(true_bands)} true vs {len(predicted_bands)} predicted"
        )
    matrix = ConfusionMatrix()
    for true_band, predicted_band in zip(true_bands, predicted_bands, strict=True):
        matrix.add(true_band, predicted_band)
    return matrix


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityBin:
    """One bucket of the reliability diagram."""

    lower_edge: float
    upper_edge: float
    sample_count: int
    mean_confidence: float
    observed_accuracy: float

    @property
    def gap(self) -> float:
        """Signed gap. Positive = overconfident, negative = underconfident."""
        return self.mean_confidence - self.observed_accuracy


@dataclass(frozen=True)
class CalibrationReport:
    """Full calibration picture for one set of predictions."""

    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    bins: tuple[ReliabilityBin, ...]
    sample_count: int

    @property
    def verdict(self) -> str:
        """Plain-language reading of the ECE.

        Thresholds follow the conventional reading of ECE in the calibration
        literature; they are a communication aid, not a standard.
        """
        if self.expected_calibration_error < 0.05:
            return "well calibrated"
        if self.expected_calibration_error < 0.15:
            return "moderately miscalibrated"
        return "severely miscalibrated"

    @property
    def direction(self) -> str:
        """Whether the system is systematically over- or under-confident.

        This matters more than the magnitude for deciding what to do next:
        overconfidence means the thresholds are letting unsafe cases through,
        underconfidence means the system is abstaining when it knows enough.
        """
        weighted_gap = sum(
            bin_.gap * bin_.sample_count for bin_ in self.bins if bin_.sample_count
        )
        total = sum(bin_.sample_count for bin_ in self.bins)
        if total == 0:
            return "no data"
        mean_gap = weighted_gap / total
        if mean_gap > 0.02:
            return "overconfident"
        if mean_gap < -0.02:
            return "underconfident"
        return "balanced"

    def render(self) -> str:
        lines = [
            f"samples                    {self.sample_count}",
            f"expected calibration error {self.expected_calibration_error:.4f}  ({self.verdict})",
            f"maximum calibration error  {self.maximum_calibration_error:.4f}",
            f"brier score                {self.brier_score:.4f}",
            f"direction                  {self.direction}",
            "",
            "reliability diagram",
            "  bin          n     mean conf    accuracy         gap",
            "  " + "-" * 52,
        ]
        for bin_ in self.bins:
            if bin_.sample_count == 0:
                continue
            marker = "  <-- overconfident" if bin_.gap > 0.10 else ""
            lines.append(
                f"  [{bin_.lower_edge:.1f},{bin_.upper_edge:.1f})"
                f"{bin_.sample_count:7d}"
                f"{bin_.mean_confidence:13.3f}"
                f"{bin_.observed_accuracy:12.3f}"
                f"{bin_.gap:+12.3f}{marker}"
            )
        return "\n".join(lines)


def compute_calibration(
    confidences: list[float],
    correctness: list[bool],
    bin_count: int = DEFAULT_BIN_COUNT,
) -> CalibrationReport:
    """Compute ECE, MCE, Brier score and reliability bins.

    Uses equal-WIDTH bins rather than equal-frequency. Equal-width is the
    standard formulation and, more importantly here, it keeps bin edges aligned
    with the policy thresholds (0.4, 0.7) so the diagram can be read directly
    against the decision table. The cost is that sparse bins are noisy, which is
    why ``sample_count`` is reported per bin rather than hidden.
    """
    if len(confidences) != len(correctness):
        raise ValueError(
            f"length mismatch: {len(confidences)} confidences vs {len(correctness)} labels"
        )
    if not confidences:
        return CalibrationReport(0.0, 0.0, 0.0, (), 0)

    total_samples = len(confidences)
    bin_width = 1.0 / bin_count

    # Group sample indices by bin. Confidence of exactly 1.0 belongs in the top
    # bin, not a phantom bin above it -- hence the explicit clamp.
    bucketed_indices: dict[int, list[int]] = defaultdict(list)
    for index, confidence in enumerate(confidences):
        bin_index = min(int(confidence / bin_width), bin_count - 1)
        bucketed_indices[bin_index].append(index)

    bins: list[ReliabilityBin] = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0

    for bin_index in range(bin_count):
        indices = bucketed_indices.get(bin_index, [])
        lower_edge = bin_index * bin_width
        upper_edge = lower_edge + bin_width

        if not indices:
            bins.append(ReliabilityBin(lower_edge, upper_edge, 0, 0.0, 0.0))
            continue

        mean_confidence = sum(confidences[i] for i in indices) / len(indices)
        observed_accuracy = sum(1 for i in indices if correctness[i]) / len(indices)
        gap = abs(mean_confidence - observed_accuracy)

        # ECE weights each bin by how much of the data it holds, so a wildly
        # wrong bin holding three samples cannot dominate the headline number.
        expected_calibration_error += (len(indices) / total_samples) * gap
        maximum_calibration_error = max(maximum_calibration_error, gap)

        bins.append(
            ReliabilityBin(
                lower_edge=lower_edge,
                upper_edge=upper_edge,
                sample_count=len(indices),
                mean_confidence=mean_confidence,
                observed_accuracy=observed_accuracy,
            )
        )

    brier_score = sum(
        (confidence - (1.0 if is_correct else 0.0)) ** 2
        for confidence, is_correct in zip(confidences, correctness, strict=True)
    ) / total_samples

    return CalibrationReport(
        expected_calibration_error=expected_calibration_error,
        maximum_calibration_error=maximum_calibration_error,
        brier_score=brier_score,
        bins=tuple(bins),
        sample_count=total_samples,
    )
