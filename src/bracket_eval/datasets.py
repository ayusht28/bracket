"""Dataset loaders for evaluation.

WHY THIS MODULE EXISTS
----------------------
The upstream eval is circular. ``scripts/generate_synthetic_chats.py`` writes
the test transcripts and ``eval_pipeline_against_synthetic.py`` then scores them
with the same lexicon that informed the generation. Reported accuracy: 100%.

That number measures one thing -- that a rule system can recover labels from
text written to contain the rules' own trigger words. It is a tautology with a
confusion matrix attached. A held-out set whose author did not know the lexicon
would score very differently, and that difference is the entire point.

This module supplies two honest alternatives.

1. HELD-OUT TRANSCRIPTS (bundled, offline)
   ``data/holdout_en.jsonl`` and ``data/holdout_hinglish.jsonl``. Written as
   ordinary chat, deliberately NOT as keyword bait. They include the cases that
   break rule systems: terse adults, teens talking about nothing in particular,
   adults discussing their own children, non-native English writers. The
   pipeline is expected to get a meaningful fraction of these WRONG, and the
   resulting number is the useful one.

2. BLOG AUTHORSHIP CORPUS (external, opt-in)
   Schler, Koppel, Argamon & Pennebaker (2006) -- ~19,000 bloggers with
   self-reported ages. Cited in the upstream lexicon's own docstring as the
   basis for its weight ordering, then never used to test it. ``load_blog_corpus``
   reads it if present and maps age to band.

   NOTE ON PROVENANCE: I cannot fetch or verify this dataset from here, and
   dataset mirrors move. Confirm the source and licence before relying on it,
   and treat the citation as a pointer rather than a verified link.

BAND BOUNDARIES
---------------
    child  <= 12
    teen   13-17
    adult  >= 18

The 17/18 boundary is legally meaningful and linguistically invisible, which is
exactly why the system emits bands rather than ages. Expect the teen/adult
confusion cell to stay populated no matter what -- that is a property of the
problem, not a bug in the pipeline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _REPO_ROOT / "data"

CHILD_MAX_AGE: int = 12
TEEN_MAX_AGE: int = 17


def band_for_age(age: int) -> str:
    """Map a numeric age to a coarse band."""
    if age <= CHILD_MAX_AGE:
        return "child"
    if age <= TEEN_MAX_AGE:
        return "teen"
    return "adult"


def load_jsonl(path: Path) -> list[dict]:
    """Load conversations from a JSON Lines file.

    Expected shape per line:
        {"id": str, "band": str, "turns": [str, ...], "note": str (optional)}

    ``note`` records WHY a case is in the set (e.g. "adult who writes tersely").
    It is carried through so failures can be grouped by failure mode rather than
    just counted -- knowing that every miss is a non-native speaker is far more
    actionable than knowing accuracy is 68%.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            f"Expected bundled data in {DATA_DIR}. Re-clone if missing."
        )

    conversations: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {error}") from error

            missing_fields = {"band", "turns"} - record.keys()
            if missing_fields:
                raise ValueError(f"{path}:{line_number} missing fields: {missing_fields}")
            conversations.append(record)

    return conversations


def load_holdout(include_hinglish: bool = True) -> list[dict]:
    """Load the bundled held-out set.

    Hinglish is included by default because excluding it would reproduce the
    exact blind spot this module exists to fix -- an eval that only measures
    English performance will happily report that abstaining on Hinglish is fine.
    """
    conversations = load_jsonl(DATA_DIR / "holdout_en.jsonl")
    if include_hinglish:
        conversations.extend(load_jsonl(DATA_DIR / "holdout_hinglish.jsonl"))
    return conversations


def load_adversarial() -> list[dict]:
    """Load the evasion suite -- minors actively claiming to be adults.

    Scored differently from the main set: the question is not "was the band
    right" but "did the pipeline ever settle on adult", because a single
    adult-classified turn is enough for a host product to open up.
    """
    return load_jsonl(DATA_DIR / "adversarial.jsonl")


def load_blog_corpus(
    csv_path: Path,
    max_rows: int | None = None,
    turns_per_conversation: int = 4,
    min_chars_per_turn: int = 40,
) -> list[dict]:
    """Load the Blog Authorship Corpus and reshape posts into conversations.

    Expected columns: ``text`` and ``age`` (case-insensitive). Extra columns are
    ignored so the various mirrors of this dataset all work.

    WHY RESHAPE POSTS INTO TURNS: the pipeline accumulates evidence ACROSS turns
    and applies a sparsity penalty below three turns. Feeding it one long blog
    post would test a code path the product never uses. Splitting each post into
    sentence-ish chunks approximates multi-turn chat closely enough to exercise
    the real accumulation logic.

    This is an approximation and should be reported as one. Blog prose is not
    chat, and the domain gap will cost accuracy. It is still enormously more
    honest than scoring against text the lexicon authored.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Blog Authorship Corpus not found at {csv_path}\n"
            "This dataset is not bundled (licence + size). Download it, then:\n"
            "  python scripts/run_eval.py --blog-corpus /path/to/blogtext.csv"
        )

    conversations: list[dict] = []

    with csv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")

        # Tolerate column-name variation across mirrors of this dataset.
        column_lookup = {name.lower().strip(): name for name in reader.fieldnames}
        text_column = column_lookup.get("text")
        age_column = column_lookup.get("age")
        if text_column is None or age_column is None:
            raise ValueError(
                f"{csv_path} needs 'text' and 'age' columns; found {reader.fieldnames}"
            )

        for row_index, row in enumerate(reader):
            if max_rows is not None and len(conversations) >= max_rows:
                break

            raw_age = (row.get(age_column) or "").strip()
            try:
                age = int(raw_age)
            except ValueError:
                continue  # unlabelled rows are dropped, not guessed

            text = (row.get(text_column) or "").strip()
            turns = _split_into_turns(text, turns_per_conversation, min_chars_per_turn)
            if len(turns) < 2:
                continue  # too short to exercise multi-turn accumulation

            conversations.append(
                {
                    "id": f"blog-{row_index}",
                    "band": band_for_age(age),
                    "turns": turns,
                    "note": f"blog corpus, self-reported age {age}",
                }
            )

    return conversations


def _split_into_turns(text: str, target_turns: int, min_chars: int) -> list[str]:
    """Chunk a block of prose into roughly turn-sized pieces.

    Splits on sentence enders, then greedily packs sentences until a chunk
    clears ``min_chars``. Greedy packing rather than even division because
    turns of wildly uneven length would skew the reading-level scorer.
    """
    normalised = " ".join(text.split())
    if not normalised:
        return []

    sentences: list[str] = []
    current = ""
    for character in normalised:
        current += character
        if character in ".!?":
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    turns: list[str] = []
    buffer = ""
    for sentence in sentences:
        buffer = f"{buffer} {sentence}".strip()
        if len(buffer) >= min_chars:
            turns.append(buffer)
            buffer = ""
        if len(turns) >= target_turns:
            break

    if buffer and len(turns) < target_turns:
        turns.append(buffer)

    return turns


def summarise(conversations: list[dict]) -> str:
    """One-line-per-band breakdown of a loaded dataset."""
    counts: dict[str, int] = {}
    for conversation in conversations:
        counts[conversation["band"]] = counts.get(conversation["band"], 0) + 1
    parts = [f"{band}={counts.get(band, 0)}" for band in ("child", "teen", "adult")]
    return f"{len(conversations)} conversations  ({'  '.join(parts)})"
