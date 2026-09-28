"""Tests for calibration and classification metrics."""

from __future__ import annotations

import pytest

from src.bracket_eval.metrics import (
    ConfusionMatrix,
    compute_calibration,
    confusion_matrix,
)


class TestConfusionMatrix:
    def test_perfect_predictions_give_accuracy_one(self) -> None:
        matrix = confusion_matrix(["child", "teen", "adult"], ["child", "teen", "adult"])
        assert matrix.accuracy == 1.0
        assert matrix.abstention_rate == 0.0

    def test_abstention_counts_as_wrong_in_strict_accuracy(self) -> None:
        """An abstention is a safe failure, but still a failure."""
        matrix = confusion_matrix(["child", "teen"], ["child", "unknown"])
        assert matrix.accuracy == 0.5
        assert matrix.accuracy_excluding_abstentions == 1.0
        assert matrix.abstention_rate == 0.5

    def test_child_as_adult_is_tracked_separately(self) -> None:
        """The worst error must be visible, not averaged into accuracy."""
        matrix = confusion_matrix(
            ["child", "child", "adult"], ["adult", "teen", "adult"]
        )
        assert matrix.child_missed_as_adult() == 1
        assert matrix.minor_missed_as_adult() == 1

    def test_minor_missed_counts_teens_too(self) -> None:
        matrix = confusion_matrix(["child", "teen"], ["adult", "adult"])
        assert matrix.minor_missed_as_adult() == 2

    def test_per_band_precision_recall(self) -> None:
        matrix = confusion_matrix(
            ["child", "child", "teen", "adult"],
            ["child", "teen", "teen", "adult"],
        )
        child_stats = matrix.per_band()["child"]
        assert child_stats["precision"] == 1.0  # 1 predicted child, 1 correct
        assert child_stats["recall"] == 0.5  # 2 actual children, 1 found

    def test_empty_matrix_does_not_divide_by_zero(self) -> None:
        matrix = ConfusionMatrix()
        assert matrix.accuracy == 0.0
        assert matrix.abstention_rate == 0.0
        assert matrix.accuracy_excluding_abstentions == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            confusion_matrix(["child"], ["child", "teen"])


class TestCalibration:
    def test_perfect_calibration_gives_near_zero_ece(self) -> None:
        """Confidence 0.95 on 95 correct out of 100 -> ECE should be tiny."""
        confidences = [0.95] * 100
        correctness = [True] * 95 + [False] * 5
        report = compute_calibration(confidences, correctness)
        assert report.expected_calibration_error < 0.02
        assert report.verdict == "well calibrated"

    def test_overconfidence_is_detected_and_named(self) -> None:
        """High confidence, low accuracy -> large ECE, flagged overconfident."""
        confidences = [0.95] * 100
        correctness = [True] * 30 + [False] * 70
        report = compute_calibration(confidences, correctness)
        assert report.expected_calibration_error > 0.5
        assert report.verdict == "severely miscalibrated"
        assert report.direction == "overconfident"

    def test_underconfidence_is_detected_and_named(self) -> None:
        """Low confidence, high accuracy -> flagged underconfident.

        This is the direction the real pipeline actually exhibits, so it must
        not be silently reported as 'miscalibrated' without a direction.
        """
        confidences = [0.15] * 100
        correctness = [True] * 90 + [False] * 10
        report = compute_calibration(confidences, correctness)
        assert report.direction == "underconfident"

    def test_confidence_of_exactly_one_lands_in_top_bin(self) -> None:
        """Guards the off-by-one that would create a phantom 11th bin."""
        report = compute_calibration([1.0], [True], bin_count=10)
        populated = [b for b in report.bins if b.sample_count > 0]
        assert len(populated) == 1
        assert populated[0].upper_edge == pytest.approx(1.0)

    def test_brier_score_rewards_confident_correctness(self) -> None:
        confident_and_right = compute_calibration([0.99] * 10, [True] * 10)
        hedging = compute_calibration([0.5] * 10, [True] * 10)
        assert confident_and_right.brier_score < hedging.brier_score

    def test_empty_input_returns_zeroed_report(self) -> None:
        report = compute_calibration([], [])
        assert report.sample_count == 0
        assert report.expected_calibration_error == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_calibration([0.5], [True, False])

    def test_max_calibration_error_catches_single_bad_bin(self) -> None:
        """One badly broken bin must surface in MCE even if ECE looks fine."""
        confidences = [0.5] * 99 + [0.95]
        correctness = [True] * 50 + [False] * 49 + [False]
        report = compute_calibration(confidences, correctness)
        assert report.maximum_calibration_error > 0.9
