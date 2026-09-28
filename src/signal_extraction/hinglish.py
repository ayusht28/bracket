"""Code-mixed Hindi-English (Hinglish) cue lexicon.

WHY THIS MODULE EXISTS
----------------------
The upstream English lexicon abstains on non-English text: ``language_detect``
sees Devanagari or a low ASCII ratio, and ``keyword_extractor`` returns an empty
SignalSet. Abstention is the correct default for languages nobody has written
cues for -- a false signal is worse than no signal.

But abstention is NOT correct for Hinglish, and Hinglish is the single largest
gap in the original system. Consider:

    English : "mom said I have to log off at 9"   -> guardian_reference, weight 0.7
    Hinglish: "mummy ne bola 9 baje tak hi"       -> ZERO cues, band unknown

Identical information, identical protective need, completely different outcome.
India is one of the largest markets for exactly the chat products this system
targets, and romanized Hinglish is the default register for teenagers there.
A child-safety signal that goes blind on that population is not a safety signal.

DESIGN CONSTRAINTS (inherited from the English lexicon, deliberately preserved)
------------------------------------------------------------------------------
1. Weights live HERE, in auditable Python -- never assigned by a model.
2. Same weight ordering: disclosure > topic/life-context > lexical style.
   Lexical style stays weak because a non-native adult and a native child look
   alike on exactly those features.
3. Both scripts for every term. Romanized Hinglish ("mummy", "papa") dominates
   in chat, but Devanagari appears in mixed input, so both map to one subtype.

WHY ROMANIZED SPELLING IS PLURAL
--------------------------------
Hinglish has no orthographic standard. "baje" / "baj" / "bje" are all common,
as are "nahi" / "nahin" / "nhi". We enumerate the frequent variants rather than
attempting phonetic normalisation, because a normaliser that is wrong is worse
than a keyword list that is incomplete -- an incomplete list abstains, a wrong
normaliser invents cues.

A DELIBERATE OMISSION
---------------------
There is no Hinglish equivalent of the "adult self-claim" detector here. Claim
phrasing in code-mixed chat is too varied to enumerate safely, and a missed
adult claim fails safe (band stays young, posture stays protective) whereas a
falsely-detected one fails open. We omit rather than guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Weight for a Hinglish subtype the lexicon does not recognise. Matches the
# English lexicon's DEFAULT_WEIGHT so the two paths score consistently.
DEFAULT_WEIGHT: float = 0.2


@dataclass(frozen=True)
class HinglishCueSpec:
    """One recognised Hinglish cue subtype.

    Mirrors ``lexicon.CueSpec`` exactly so both lexicons can feed the same
    downstream Cue construction without any special-casing in the extractor.
    """

    cue_type: str  # disclosure | topic | vocab | style | reading_level
    weight: float  # deterministic weight in [0, 1]
    band_hint: str  # "child" | "teen" | "adult" | ""
    keywords: tuple[str, ...]  # lowercased substrings, romanized + Devanagari


# ---------------------------------------------------------------------------
# Cue specifications
#
# Ordering within this dict is the scan order, so the strongest and least
# ambiguous cues are matched first.
# ---------------------------------------------------------------------------

HINGLISH_CUE_SPECS: dict[str, HinglishCueSpec] = {
    # --- explicit education-stage disclosure (strongest, mirrors grade_level) ---
    # Indian school stages are named by ordinal class number. "10th"/"12th" are
    # board-exam years and are extremely high-signal for 15-17 year olds.
    "class_level": HinglishCueSpec(
        "disclosure", 0.9, "teen",
        (
            "6th class", "7th class", "8th class", "9th class", "10th class",
            "11th class", "12th class",
            "class 6", "class 7", "class 8", "class 9", "class 10",
            "class 11", "class 12",
            "9th standard", "10th standard", "11th standard", "12th standard",
            "board exam", "board exams", "boards ki", "boards hai",
            "cbse", "icse", "ssc board", "hsc",
            "कक्षा", "बोर्ड परीक्षा",
        ),
    ),
    # Primary-stage terms. Separate subtype from class_level because the band
    # hint differs -- these indicate child, not teen.
    "primary_school_hinglish": HinglishCueSpec(
        "disclosure", 0.9, "child",
        (
            "1st class", "2nd class", "3rd class", "4th class", "5th class",
            "class 1", "class 2", "class 3", "class 4", "class 5",
            "nursery", "kg mein", "chhutti ho gayi", "chutti ho gayi",
            "प्राथमिक", "नर्सरी",
        ),
    ),
    # --- guardian / family reference (strong topic cue) ---
    # The highest-frequency Hinglish child signal in practice. Kinship terms are
    # used constantly in Indian chat and are near-absent from adult peer talk
    # about their own permissions.
    "guardian_hinglish": HinglishCueSpec(
        "topic", 0.7, "child",
        (
            "mummy", "mumma", "mammi", "papa ne", "papa bol", "pappa",
            "mummy ne", "mummy bol", "mummy ko", "mummy se",
            "ghar pe daant", "daant padegi", "daant padi",
            "ghar walon", "ghar wale", "gharwale",
            "bade bhaiya", "badi didi",
            "मम्मी", "पापा", "घरवाले",
        ),
    ),
    # --- curfew / permission (strong topic cue) ---
    # "permission needed to be online" is the single most reliable minor signal
    # across languages. This is the Hinglish surface form of it.
    "curfew_hinglish": HinglishCueSpec(
        "topic", 0.7, "child",
        (
            "baje tak", "baj tak", "bje tak",
            "permission chahiye", "permission nahi", "permission nhi",
            "phone chhin", "phone chin liya", "phone le liya",
            "sone ka time", "sona padega", "so jana padega",
            "allowed nahi", "allowed nhi", "mana kiya",
            "बजे तक", "इजाज़त",
        ),
    ),
    # --- school-life topic (strong, robust to gaming) ---
    "school_life_hinglish": HinglishCueSpec(
        "topic", 0.65, "teen",
        (
            "tuition", "tution", "coaching class", "coaching jana",
            "homework nahi", "hw nahi", "hw nhi", "homework karna",
            "school jana", "school gaya", "school mein", "skul",
            "pt sir", "maths sir", "science mam", "english mam",
            "class teacher", "principal ne", "unit test", "half yearly",
            "exam ki tayari", "syllabus khatam",
            "स्कूल", "ट्यूशन", "परीक्षा",
        ),
    ),
    # --- adult life-context (counter-signal, prevents one-sided scoring) ---
    # WHY: without adult Hinglish cues the module could only ever push toward
    # "child", which would bias every Hinglish speaker young. Adult life-stage
    # markers must be detectable in the same register.
    "adult_life_hinglish": HinglishCueSpec(
        "topic", 0.65, "adult",
        (
            "office mein", "office jana", "office se", "meeting hai",
            "salary", "emi", "home loan", "loan ki", "gst",
            "biwi", "husband ne", "shaadi ho gayi", "meri wife",
            "mere bacche", "beta school", "beti school",
            "landlord", "rent dena", "maid nahi aayi",
            "ऑफिस", "तनख्वाह", "शादी",
        ),
    ),
    # --- juvenile lexical register (WEAK, matches English style weighting) ---
    # Deliberately 0.25. These are cohort-drifting slang markers -- adults use
    # them too, and they carry class/region bias. They nudge; they never
    # establish a band (STRONG_CUE_TYPES excludes "style" downstream).
    "juvenile_register_hinglish": HinglishCueSpec(
        "style", 0.25, "teen",
        (
            "bhai sun", "bhai please", "yaar please", "plss", "plzz",
            "kitna maza", "bahut maza", "mast hai", "ekdum mast",
            "op hai", "noob", "bc yaar", "arre yaar",
        ),
    ),
}

# Scan order -- dict insertion order, made explicit so behaviour cannot change
# silently if the dict is reordered during an edit.
_SUBTYPE_ORDER: tuple[str, ...] = tuple(HINGLISH_CUE_SPECS.keys())


# ---------------------------------------------------------------------------
# Hinglish detection
# ---------------------------------------------------------------------------

# Function words that are near-unique to Hindi/Hinglish and extremely common.
# WHY function words rather than content words: they appear in almost every
# Hinglish sentence regardless of topic, so detection does not depend on the
# sentence happening to be about school or family.
#
# The set is SPLIT because some romanized Hindi function words are spelled
# identically to common English words -- "the", "main", "hi", "par", "se", "ka".
# A flat count over the combined set misfires on ordinary English: "the main
# point" contains two markers and would be called Hinglish. That is not a
# hypothetical; it was caught by test_single_ambiguous_marker_does_not_trigger.
_UNAMBIGUOUS_MARKERS: frozenset[str] = frozenset({
    "hai", "hain", "tha", "thi", "nahi", "nahin", "nhi", "kya", "kyu", "kyun",
    "aur", "lekin", "bhi", "toh", "tho", "matlab", "mein",
    "karna", "karne", "karta", "karti", "gaya", "gayi", "raha", "rahi",
    "mera", "meri", "tera", "teri", "apna", "apni", "unka", "iska",
    "bola", "boli", "diya", "liya", "hua", "hui", "jana",
    "yaar", "bhai", "acha", "accha", "theek", "thik", "abhi", "phir",
    "kuch", "bahut", "thoda", "jaldi", "wapas", "chalo", "arre",
})

# Spelled the same as English words. These only corroborate -- they can never
# trigger detection on their own, no matter how many appear.
_AMBIGUOUS_MARKERS: frozenset[str] = frozenset({
    "the", "main", "hi", "par", "se", "ka", "ki", "ke", "ko", "bol", "he",
})

# One unambiguous marker is decisive: no English sentence contains "nahi" or
# "kyun" by accident. Ambiguous markers are counted only as a secondary route
# and need a higher bar.
_MIN_UNAMBIGUOUS_MARKERS: int = 1
_MIN_AMBIGUOUS_MARKERS_ALONE: int = 4

# Devanagari Unicode block. Any character here is decisive on its own -- no
# English text contains Devanagari by accident.
_DEVANAGARI_PATTERN = re.compile(r"[ऀ-ॿ]")

_WORD_PATTERN = re.compile(r"[a-z]+")


def is_hinglish(text: str) -> bool:
    """Return True when *text* looks like code-mixed Hindi-English.

    Three independent triggers, in order of reliability:
      1. Any Devanagari character -- decisive, zero false-positive risk.
      2. One or more UNAMBIGUOUS romanized markers ("nahi", "kyun", "hai").
         These do not collide with English, so one is enough.
      3. Four or more AMBIGUOUS markers ("the", "main", "ka"). This route
         exists for heavily-transliterated text that happens to avoid the
         unambiguous list, and the high bar keeps ordinary English out.

    The split is what stops "the main point" being classified as Hinglish.
    """
    if _DEVANAGARI_PATTERN.search(text):
        return True

    words_in_text = set(_WORD_PATTERN.findall(text.lower()))

    unambiguous_hits = words_in_text & _UNAMBIGUOUS_MARKERS
    if len(unambiguous_hits) >= _MIN_UNAMBIGUOUS_MARKERS:
        return True

    ambiguous_hits = words_in_text & _AMBIGUOUS_MARKERS
    return len(ambiguous_hits) >= _MIN_AMBIGUOUS_MARKERS_ALONE


# ---------------------------------------------------------------------------
# Lookup helpers -- same surface API as lexicon.py so the extractor can call
# either module without branching on which lexicon it is talking to.
# ---------------------------------------------------------------------------


def assign_weight(subtype: str) -> float:
    """Return the deterministic weight for a Hinglish *subtype*."""
    spec = HINGLISH_CUE_SPECS.get(subtype)
    return spec.weight if spec is not None else DEFAULT_WEIGHT


def band_hint(subtype: str) -> str:
    """Return the band lean for a Hinglish *subtype*, or '' if unknown."""
    spec = HINGLISH_CUE_SPECS.get(subtype)
    return spec.band_hint if spec is not None else ""


def cue_type_for(subtype: str) -> str:
    """Return the Cue.type for a Hinglish *subtype*, or '' if unknown."""
    spec = HINGLISH_CUE_SPECS.get(subtype)
    return spec.cue_type if spec is not None else ""


def is_known_subtype(subtype: str) -> bool:
    """Return True when *subtype* is defined in the Hinglish lexicon."""
    return subtype in HINGLISH_CUE_SPECS


def classify_text(text: str) -> list[tuple[str, str, str]]:
    """Scan *text* for Hinglish age cues.

    Returns (cue_type, subtype, matched_snippet) tuples, at most one per
    subtype, in deterministic scan order -- identical contract to
    ``lexicon.classify_text`` so results from both can be concatenated.

    Returns an empty list when the text is not Hinglish, so calling this on
    pure English input is safe and cheap.
    """
    if not is_hinglish(text):
        return []

    lowered = text.lower()
    found: list[tuple[str, str, str]] = []

    for subtype in _SUBTYPE_ORDER:
        spec = HINGLISH_CUE_SPECS[subtype]
        for keyword in spec.keywords:
            if keyword in lowered:
                found.append((spec.cue_type, subtype, keyword.strip()))
                break  # at most one match per subtype, matching English lexicon

    return found
