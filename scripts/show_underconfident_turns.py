#!/usr/bin/env python3
"""Show the individual turns behind the underconfidence finding.

WHY THIS SCRIPT EXISTS
----------------------
``run_eval.py`` reports the reliability diagram, and one row of it carries the
whole argument:

    bin          n     mean conf    accuracy         gap
    [0.3,0.4)   21        0.336       0.952      -0.617

That row says: 21 turns where the system scored itself ~34% sure, and it was
right on 20 of them. The claim "the system is underconfident, and the
abstention rate is a symptom of that" rests entirely on rows like this one.

An aggregate is easy to distrust, so this script drops to the individual turns
and prints them. Every number printed here is recomputed from the pipeline on
the spot -- nothing is read from a cached report.

Usage:
    python scripts/show_underconfident_turns.py
    python scripts/show_underconfident_turns.py --low 0.2 --high 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.bracket_eval import datasets  # noqa: E402
from src.bracket_eval.pipeline import replay_dataset  # noqa: E402
from src.policy_decision.table import lookup as policy_lookup  # noqa: E402

# The cut points from src/policy_decision/table.py. A turn scoring below
# LOW_THRESHOLD lands in the "low" confidence bucket, which for a minor means
# "caution" rather than a real restriction -- the signal is effectively parked.
LOW_THRESHOLD = 0.4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=float, default=0.3,
                        help="Lower edge of the confidence band to inspect")
    parser.add_argument("--high", type=float, default=0.4,
                        help="Upper edge of the confidence band to inspect")
    args = parser.parse_args()

    results = replay_dataset(datasets.load_holdout(include_hinglish=True))

    # Collect every TURN whose confidence falls in the band under inspection.
    # A turn is "correct" when the band it proposed matches the conversation's
    # true label -- the same definition run_eval.py uses for calibration.
    inspected: list[tuple[str, int, str, str, float, bool]] = []
    for result in results:
        for turn in result.turns:
            if args.low <= turn.confidence < args.high:
                inspected.append((
                    result.conversation_id,
                    turn.turn_index,
                    result.true_band,
                    turn.band,
                    turn.confidence,
                    turn.band == result.true_band,
                ))

    if not inspected:
        print(f"No turns scored between {args.low} and {args.high}.")
        return 0

    correct = sum(1 for *_, ok in inspected if ok)
    total = len(inspected)

    print("=" * 78)
    print(f"TURNS THE SYSTEM SCORED {args.low:.2f}-{args.high:.2f} CONFIDENT")
    print("=" * 78)
    print(f"{'conversation':14}{'turn':6}{'true':8}{'predicted':11}"
          f"{'conf':7}{'right?':8}{'posture it triggered'}")
    print("-" * 78)

    for conv_id, turn_index, true_band, predicted, confidence, ok in inspected:
        # Re-derive the posture through the REAL policy table, so the printed
        # consequence is the one the product would actually have applied.
        decision = policy_lookup(predicted, confidence)
        print(f"{conv_id:14}{turn_index:<6}{true_band:8}{predicted:11}"
              f"{confidence:<7.3f}{'YES' if ok else 'no':8}"
              f"{decision.posture_level} / {decision.action}")

    print("-" * 78)
    print(f"correct: {correct}/{total}  =  {correct / total:.1%}")
    print(f"the system's own stated confidence on these turns: "
          f"~{sum(c for *_, c, _ in inspected) / total:.1%}")
    print()

    gap = (correct / total) - (sum(c for *_, c, _ in inspected) / total)
    print(f"It was right {correct / total:.1%} of the time while claiming to be "
          f"{sum(c for *_, c, _ in inspected) / total:.1%} sure.")
    print(f"Underconfidence gap: {gap:+.1%}")
    print()

    below_cut = [x for x in inspected if x[4] < LOW_THRESHOLD]
    if below_cut:
        correct_below = sum(1 for *_, ok in below_cut if ok)
        print(f"Of these, {len(below_cut)} scored below the {LOW_THRESHOLD} cut point "
              f"in policy_decision/table.py,")
        print(f"and {correct_below} of those {len(below_cut)} were CORRECT — "
              f"right answers parked as 'low confidence'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
