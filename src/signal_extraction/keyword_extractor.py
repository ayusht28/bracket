"""Deterministic offline signal extractor (M2 fallback).

Builds a ``SignalSet`` from raw turn text using the lexicon keyword patterns
(``lexicon.classify_text``) plus the Flesch-Kincaid reading-level scorer. Pure
Python — no LLM, no I/O. Used when no model endpoint is configured so the whole
pipeline runs end to end without a GPU.

Weights come exclusively from the lexicon (never hand-set here), keeping the
"confidence is deterministic" invariant true.

Non-English abstention (Phase 1):
    This extractor's lexicon is English-only. Scanning non-English text produces
    false signals worse than "no evidence" (e.g. a Spanish sentence accidentally
    matching an English keyword). ``extract_cues`` therefore checks the language
    and returns an **empty SignalSet** for confidently non-English input, making
    the abstention explicit rather than an accidental side-effect.
    Undetermined / short text (language == "") is passed through normally.
"""

from __future__ import annotations

import logging

from src.contracts.models import Cue, SignalSet
from src.signal_extraction import hinglish, lexicon
from src.signal_extraction.language_detect import is_english_or_unknown
from src.signal_extraction.maturity import extract_maturity_cues
from src.signal_extraction.reading_level import compute_reading_level

logger = logging.getLogger(__name__)

# Reading level at/above this normalised score reads as adult; below reads young.
_READING_LEVEL_ADULT_THRESHOLD: float = 0.6


def extract_cues(text: str) -> SignalSet:
    """Extract age-relevant cues from *text* deterministically.

    Returns an empty SignalSet for confidently non-English text — the English
    lexicon cannot reliably score non-English input and must not produce false
    age signals from it.
    """
    # Hinglish is checked BEFORE the English abstention gate. Romanized Hinglish
    # has a high ASCII ratio, so language_detect classifies it as "en" and the
    # English lexicon scans it -- finding almost nothing, because the cues are
    # in Hindi. Devanagari-mixed Hinglish takes the opposite path and abstains
    # entirely. Both outcomes lose a real signal, so we intercept here and run
    # the dedicated Hinglish lexicon instead.
    if hinglish.is_hinglish(text):
        return _extract_hinglish_cues(text)

    if not is_english_or_unknown(text):
        logger.debug(
            "keyword_extractor: non-English text detected — abstaining "
            "(returning empty SignalSet to avoid false signals from English lexicon)"
        )
        return SignalSet(cues=[])

    cues: list[Cue] = []

    for cue_type, subtype, matched in lexicon.classify_text(text):
        cues.append(
            Cue(
                type=cue_type,  # type: ignore[arg-type]  # lexicon yields valid literals
                value=f"{subtype}: {matched!r}",
                weight=lexicon.assign_weight_any(subtype),
                subtype=subtype,
            )
        )

    if text.strip():
        rl = compute_reading_level(text)
        subtype = (
            "reading_level_high"
            if rl >= _READING_LEVEL_ADULT_THRESHOLD
            else "reading_level_low"
        )
        cues.append(
            Cue(
                type="reading_level",
                value=f"fk_normalised={rl:.2f}",
                weight=lexicon.assign_weight_any(subtype),
                subtype=subtype,
            )
        )

    # Maturity cues: weak nudge (weight 0.3, excluded from _STRONG_TYPES).
    cues.extend(extract_maturity_cues(text))

    return SignalSet(cues=cues)


def _extract_hinglish_cues(text: str) -> SignalSet:
    """Build a SignalSet from code-mixed Hindi-English text.

    MERGES both lexicons rather than replacing the English one. An earlier
    version of this function returned ONLY Hinglish cues, which was wrong for
    the obvious reason: code-mixed text is code-mixed. "abhi homework bhi baaki
    hai" carries an English school cue that the Hinglish lexicon does not
    enumerate, and swapping lexicons silently dropped it. Measured on the
    held-out set, that made the Hinglish path score LOWER confidence than the
    English fallback it replaced.

    Cue precedence when both lexicons match the same concept: the Hinglish cue
    wins, because its weight was set against code-mixed text specifically. The
    English cue is skipped rather than double-counted -- two cues for one piece
    of evidence would inflate the corroboration score, which is exactly the
    kind of silent confidence inflation this architecture exists to prevent.

    Deliberately does NOT run the Flesch-Kincaid reading-level scorer. FK is
    calibrated on English syllable structure; romanized Hindi words break its
    assumptions and it reports spuriously low scores, which would tag every
    Hinglish adult as a low-reading-level writer. Since reading_level is a weak
    non-band-establishing cue anyway, dropping it costs almost nothing and
    removes a real source of demographic bias.
    """
    cues: list[Cue] = []
    hinglish_cue_types: set[str] = set()

    # Hinglish cues first -- they take precedence on overlap.
    for cue_type, subtype, matched in hinglish.classify_text(text):
        cues.append(
            Cue(
                type=cue_type,  # type: ignore[arg-type]  # hinglish yields valid literals
                value=f"{subtype}: {matched!r}",
                weight=hinglish.assign_weight(subtype),
                subtype=subtype,
            )
        )
        hinglish_cue_types.add(cue_type)

    # Then English cues the Hinglish lexicon did not already cover. Embedded
    # English is the norm in Hinglish chat ("homework", "tuition", "office"),
    # so skipping this path throws away real signal.
    for cue_type, subtype, matched in lexicon.classify_text(text):
        if cue_type in hinglish_cue_types:
            continue  # same evidence already captured, do not double-count
        cues.append(
            Cue(
                type=cue_type,  # type: ignore[arg-type]
                value=f"{subtype}: {matched!r}",
                weight=lexicon.assign_weight_any(subtype),
                subtype=subtype,
            )
        )

    logger.debug(
        "keyword_extractor: Hinglish path produced %d cues (%d Hinglish types)",
        len(cues),
        len(hinglish_cue_types),
    )
    return SignalSet(cues=cues)
