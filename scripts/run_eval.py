#!/usr/bin/env python3
"""Run the honest evaluation: held-out accuracy, calibration, threshold fit.

Replaces the circular synthetic eval. Everything here runs on CPU with no model
and no network, so it is fast enough to sit in CI.

Usage:
    python scripts/run_eval.py                          # bundled held-out set
    python scripts/run_eval.py --no-hinglish            # English only
    python scripts/run_eval.py --blog-corpus blog.csv   # external real-age data
    python scripts/run_eval.py --fit-thresholds         # sweep policy cut points
    python scripts/run_eval.py --json report.json       # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# --no-hinglish must take effect BEFORE keyword_extractor is imported, so the
# ablation is clean. Parsed here rather than via argparse for that reason.
if "--no-hinglish" in sys.argv:
    from src.signal_extraction import hinglish as _hinglish

    _hinglish.is_hinglish = lambda _text: False  # type: ignore[assignment]

from src.bracket_eval import datasets  # noqa: E402
from src.bracket_eval.calibration import (  # noqa: E402
    fit_thresholds,
    render_comparison,
    score_thresholds,
)
from src.bracket_eval.metrics import compute_calibration, confusion_matrix  # noqa: E402
from src.bracket_eval.pipeline import replay_dataset  # noqa: E402

# Upstream's hardcoded values, kept here purely as the comparison baseline.
UPSTREAM_LOW_THRESHOLD: float = 0.4
UPSTREAM_HIGH_THRESHOLD: float = 0.7

SECTION_WIDTH = 72


def section(title: str) -> None:
    print(f"\n{'=' * SECTION_WIDTH}\n{title}\n{'=' * SECTION_WIDTH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blog-corpus", type=Path, default=None,
                        help="Path to Blog Authorship Corpus CSV (text, age columns)")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Cap on blog-corpus conversations (default 2000)")
    parser.add_argument("--no-hinglish", action="store_true",
                        help="TRUE ABLATION: disable the Hinglish lexicon but keep "
                             "every conversation, so both runs score the same data")
    parser.add_argument("--fit-thresholds", action="store_true",
                        help="Grid-search the policy confidence thresholds")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write a machine-readable report to this path")
    parser.add_argument("--show-failures", action="store_true",
                        help="Print every misclassified conversation with its note")
    arguments = parser.parse_args()

    # ---- load ----------------------------------------------------------
    if arguments.blog_corpus is not None:
        conversations = datasets.load_blog_corpus(
            arguments.blog_corpus, max_rows=arguments.max_rows
        )
        dataset_name = f"Blog Authorship Corpus ({arguments.blog_corpus.name})"
    else:
        # Always load the FULL set. The ablation disables the lexicon, not the
        # data -- otherwise the comparison is between two different datasets.
        conversations = datasets.load_holdout(include_hinglish=True)
        dataset_name = (
            "bundled held-out set  [HINGLISH LEXICON DISABLED — true ablation]"
            if arguments.no_hinglish
            else "bundled held-out set"
        )

    section(f"DATASET  —  {dataset_name}")
    print(datasets.summarise(conversations))

    # ---- replay --------------------------------------------------------
    results = replay_dataset(conversations)
    note_by_id = {c.get("id"): c.get("note", "") for c in conversations}

    true_bands = [r.true_band for r in results]
    predicted_bands = [r.predicted_band for r in results]
    matrix = confusion_matrix(true_bands, predicted_bands)

    section("BAND ACCURACY  —  held-out, not self-generated")
    print(matrix.render())
    print()
    print(f"accuracy (abstention counts as wrong)   {matrix.accuracy:.3f}")
    print(f"accuracy (committed predictions only)   {matrix.accuracy_excluding_abstentions:.3f}")
    print(f"abstention rate                         {matrix.abstention_rate:.3f}")
    print()
    print("per band:")
    for band, stats in matrix.per_band().items():
        print(f"  {band:8} precision={stats['precision']:.3f}  "
              f"recall={stats['recall']:.3f}  f1={stats['f1']:.3f}  "
              f"n={int(stats['support'])}")
    print()
    print(f"UNSAFE ERRORS — children classed adult   {matrix.child_missed_as_adult()}")
    print(f"UNSAFE ERRORS — any minor classed adult  {matrix.minor_missed_as_adult()}")

    # ---- calibration ---------------------------------------------------
    # One sample per TURN, not per conversation. Confidence gates a posture on
    # every turn, so every turn is a decision point that needs to be calibrated.
    turn_confidences: list[float] = []
    turn_correctness: list[bool] = []
    for result in results:
        for turn in result.turns:
            turn_confidences.append(turn.confidence)
            turn_correctness.append(turn.band == result.true_band)

    calibration = compute_calibration(turn_confidences, turn_correctness)

    section("CALIBRATION  —  does confidence 0.7 mean right 70% of the time?")
    print(calibration.render())

    # ---- adversarial ---------------------------------------------------
    adversarial_summary: dict = {}
    if arguments.blog_corpus is None:
        adversarial = datasets.load_adversarial()
        adversarial_results = replay_dataset(adversarial)
        ever_adult = [r for r in adversarial_results if r.ever_predicted_adult]
        flagged = [r for r in adversarial_results if any(t.evasion_flag for t in r.turns)]

        section("EVASION SUITE  —  minors claiming adult status")
        print(f"{'case':12}{'true':8}{'final':10}{'ever adult':12}{'evasion flagged':16}")
        print("-" * 58)
        for result in adversarial_results:
            was_flagged = any(turn.evasion_flag for turn in result.turns)
            print(f"{result.conversation_id:12}{result.true_band:8}"
                  f"{result.predicted_band:10}"
                  f"{'YES — FAIL' if result.ever_predicted_adult else 'no':12}"
                  f"{'yes' if was_flagged else 'no':16}")
        print()
        print(f"attack success rate (ever reached adult)  "
              f"{len(ever_adult)}/{len(adversarial_results)}")
        print(f"evasion detection rate                    "
              f"{len(flagged)}/{len(adversarial_results)}")
        adversarial_summary = {
            "total": len(adversarial_results),
            "attack_success": len(ever_adult),
            "evasion_flagged": len(flagged),
        }

    # ---- threshold fitting ---------------------------------------------
    fitted_summary: dict = {}
    if arguments.fit_thresholds:
        baseline = score_thresholds(results, UPSTREAM_LOW_THRESHOLD, UPSTREAM_HIGH_THRESHOLD)
        best, all_candidates = fit_thresholds(results)

        section("THRESHOLD FIT  —  replacing the hardcoded 0.4 / 0.7")
        print(render_comparison(best, baseline))
        print()
        print("top 5 candidates by weighted cost:")
        for candidate in all_candidates[:5]:
            print(f"  low={candidate.low_threshold:.2f}  high={candidate.high_threshold:.2f}"
                  f"  cost={candidate.total_cost:.3f}"
                  f"  acc={candidate.accuracy:.3f}"
                  f"  minors_as_adult={candidate.minors_treated_as_adult}")
        print()
        # A flat optimum means the thresholds barely matter on this data; a
        # sharp one means they are load-bearing and need more data behind them.
        cost_spread = all_candidates[-1].total_cost - all_candidates[0].total_cost
        print(f"cost spread across search space: {cost_spread:.3f}"
              f"  ({'thresholds are load-bearing' if cost_spread > 0.5 else 'optimum is flat — thresholds matter little here'})")
        fitted_summary = {"baseline": asdict(baseline), "fitted": asdict(best)}

    # ---- failure inspection --------------------------------------------
    if arguments.show_failures:
        section("MISCLASSIFIED  —  grouped for failure-mode analysis")
        for result in results:
            if result.predicted_band != result.true_band:
                note = note_by_id.get(result.conversation_id, "")
                print(f"  {result.conversation_id:12} true={result.true_band:7} "
                      f"pred={result.predicted_band:8} conf={result.final_confidence:.2f}")
                if note:
                    print(f"               {note}")

    # ---- machine-readable ----------------------------------------------
    if arguments.json is not None:
        report = {
            "dataset": dataset_name,
            "conversations": len(results),
            "accuracy": matrix.accuracy,
            "accuracy_excluding_abstentions": matrix.accuracy_excluding_abstentions,
            "abstention_rate": matrix.abstention_rate,
            "per_band": matrix.per_band(),
            "child_missed_as_adult": matrix.child_missed_as_adult(),
            "minor_missed_as_adult": matrix.minor_missed_as_adult(),
            "calibration": {
                "ece": calibration.expected_calibration_error,
                "mce": calibration.maximum_calibration_error,
                "brier": calibration.brier_score,
                "verdict": calibration.verdict,
                "direction": calibration.direction,
            },
            "adversarial": adversarial_summary,
            "thresholds": fitted_summary,
        }
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {arguments.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
