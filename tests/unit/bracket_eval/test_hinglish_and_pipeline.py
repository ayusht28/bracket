"""Tests for the Hinglish lexicon, extractor integration, and eval driver."""

from __future__ import annotations

from src.bracket_eval.calibration import fit_thresholds, score_thresholds
from src.bracket_eval.pipeline import replay_conversation, replay_dataset
from src.signal_extraction import hinglish, lexicon
from src.signal_extraction.keyword_extractor import extract_cues


class TestHinglishDetection:
    def test_devanagari_is_decisive_on_its_own(self) -> None:
        assert hinglish.is_hinglish("मम्मी ने बोला")

    def test_romanized_hinglish_detected_via_function_words(self) -> None:
        assert hinglish.is_hinglish("mummy ne bola 9 baje tak hai")

    def test_plain_english_is_not_hinglish(self) -> None:
        assert not hinglish.is_hinglish(
            "I need to review the quarterly report before Friday"
        )

    def test_single_ambiguous_marker_does_not_trigger(self) -> None:
        """'main', 'the' and 'hi' are English words too.

        Ambiguous markers must never trigger detection on their own, or
        ordinary English starts matching. This caught a real false positive.
        """
        assert not hinglish.is_hinglish("hi there")
        assert not hinglish.is_hinglish("the main point")

    def test_empty_text_is_not_hinglish(self) -> None:
        assert not hinglish.is_hinglish("")


class TestHinglishCues:
    def test_guardian_and_curfew_both_fire(self) -> None:
        cues = hinglish.classify_text("mummy ne bola 9 baje tak hi phone milega")
        subtypes = {subtype for _, subtype, _ in cues}
        assert "guardian_hinglish" in subtypes
        assert "curfew_hinglish" in subtypes

    def test_adult_cues_exist_so_hinglish_is_not_biased_young(self) -> None:
        """Without adult cues the module could only ever push toward 'child'."""
        cues = hinglish.classify_text("office mein meeting hai aur EMI bhi bharni hai")
        subtypes = {subtype for _, subtype, _ in cues}
        assert "adult_life_hinglish" in subtypes

    def test_weight_ordering_matches_english_lexicon_policy(self) -> None:
        """disclosure > topic > style, same fairness rule as upstream."""
        disclosure = hinglish.assign_weight("class_level")
        topic = hinglish.assign_weight("guardian_hinglish")
        style = hinglish.assign_weight("juvenile_register_hinglish")
        assert disclosure > topic > style

    def test_english_text_yields_no_hinglish_cues(self) -> None:
        assert hinglish.classify_text("my teacher gave us homework") == []

    def test_unknown_subtype_falls_back_to_default_weight(self) -> None:
        assert hinglish.assign_weight("not_a_real_subtype") == hinglish.DEFAULT_WEIGHT
        assert hinglish.band_hint("not_a_real_subtype") == ""


class TestExtractorIntegration:
    def test_hinglish_routed_away_from_english_lexicon(self) -> None:
        signal_set = extract_cues("mummy ne bola 9 baje tak hi")
        subtypes = {cue.subtype for cue in signal_set.cues}
        assert "guardian_hinglish" in subtypes

    def test_hinglish_path_skips_reading_level(self) -> None:
        """Flesch-Kincaid is English-calibrated; running it on romanized Hindi
        would tag every Hinglish adult as a low-reading-level writer."""
        signal_set = extract_cues("office mein meeting hai aur salary aani hai")
        cue_types = {cue.type for cue in signal_set.cues}
        assert "reading_level" not in cue_types

    def test_english_path_is_unchanged(self) -> None:
        """Regression guard: the Hinglish hook must not alter English behaviour."""
        signal_set = extract_cues("I am in 8th grade and have homework")
        subtypes = {cue.subtype for cue in signal_set.cues}
        assert "grade_level" in subtypes

    def test_hinglish_subtypes_resolve_through_lexicon_any_helpers(self) -> None:
        """Band hints must survive the trip into the rule estimator."""
        assert lexicon.band_hint_any("guardian_hinglish") == "child"
        assert lexicon.band_hint_any("adult_life_hinglish") == "adult"
        assert lexicon.assign_weight_any("curfew_hinglish") == 0.7


class TestPipelineDriver:
    def test_replay_returns_one_result_per_turn(self) -> None:
        result = replay_conversation(["hello", "how are you", "bye"], true_band="adult")
        assert len(result.turns) == 3
        assert result.turns[0].turn_index == 0

    def test_sessions_are_isolated_from_each_other(self) -> None:
        """Evidence leaking between conversations would inflate accuracy."""
        first = replay_conversation(
            ["I am in 8th grade", "homework is due"], true_band="teen"
        )
        second = replay_conversation(["quarterly revenue targets"], true_band="adult")
        assert (
            second.predicted_band != first.predicted_band
            or second.final_confidence == 0.0
        )

    def test_confidence_stays_in_unit_interval(self) -> None:
        result = replay_conversation(
            ["mom says 9pm", "I am in 8th grade", "homework", "teacher said"],
            true_band="teen",
        )
        for turn in result.turns:
            assert 0.0 <= turn.confidence <= 1.0

    def test_ever_predicted_adult_catches_mid_conversation_slip(self) -> None:
        """A band that touches 'adult' even once has already failed."""
        result = replay_conversation(
            ["I am 28 years old", "my mom says I have to do homework"],
            true_band="child",
        )
        assert result.ever_predicted_adult is True

    def test_replay_dataset_preserves_ids(self) -> None:
        results = replay_dataset(
            [{"id": "abc", "band": "adult", "turns": ["hello there friend"]}]
        )
        assert results[0].conversation_id == "abc"


class TestThresholdFitting:
    def test_scoring_penalises_minors_treated_as_adult(self) -> None:
        """The asymmetric cost function must actually be asymmetric."""
        minor_as_adult = replay_dataset(
            [{"id": "x", "band": "child",
              "turns": ["quarterly revenue targets and EBITDA"]}]
        )
        adult_ok = replay_dataset(
            [{"id": "y", "band": "adult",
              "turns": ["quarterly revenue targets and EBITDA"]}]
        )
        bad = score_thresholds(minor_as_adult, 0.4, 0.7)
        good = score_thresholds(adult_ok, 0.4, 0.7)
        assert bad.total_cost >= good.total_cost

    def test_fit_respects_minimum_separation(self) -> None:
        """A degenerate low==high would delete the caution tier entirely."""
        results = replay_dataset(
            [{"id": "a", "band": "teen", "turns": ["I am in 8th grade", "homework"]}]
        )
        _, candidates = fit_thresholds(results, step=0.05, minimum_separation=0.10)
        for candidate in candidates:
            assert candidate.high_threshold - candidate.low_threshold >= 0.10 - 1e-9

    def test_fit_is_deterministic_across_runs(self) -> None:
        results = replay_dataset(
            [{"id": "a", "band": "teen", "turns": ["I am in 8th grade", "homework"]}]
        )
        first, _ = fit_thresholds(results)
        second, _ = fit_thresholds(results)
        assert first.low_threshold == second.low_threshold
        assert first.high_threshold == second.high_threshold


class TestHinglishMergesWithEnglishLexicon:
    """Regression guards for the merge fix.

    An earlier version of _extract_hinglish_cues returned ONLY Hinglish cues,
    which silently dropped English cues embedded in code-mixed text. Measured,
    that made the Hinglish path score lower confidence than the English
    fallback it replaced.
    """

    def test_english_cue_inside_hinglish_text_is_kept(self) -> None:
        """'homework' is English and not in the Hinglish keyword list."""
        signal_set = extract_cues("abhi homework bhi baaki hai")
        subtypes = {cue.subtype for cue in signal_set.cues}
        assert "school_topic" in subtypes, (
            "English cue dropped from code-mixed text — the Hinglish path is "
            "replacing the English lexicon instead of merging with it"
        )

    def test_hinglish_cue_wins_on_overlap_without_double_counting(self) -> None:
        """One piece of evidence must not produce two cues.

        Double-counting would inflate the corroboration score, which is exactly
        the silent confidence inflation this architecture exists to prevent.
        """
        signal_set = extract_cues("mummy ne bola 9 baje tak hi")
        topic_cues = [cue for cue in signal_set.cues if cue.type == "topic"]
        hinglish_topics = [c for c in topic_cues if c.subtype.endswith("_hinglish")]
        english_topics = [c for c in topic_cues if not c.subtype.endswith("_hinglish")]
        assert hinglish_topics, "Hinglish topic cue should fire"
        assert not english_topics, "English topic cue should be skipped on overlap"

    def test_hindi_dominant_text_still_establishes_a_band(self) -> None:
        """No English loanword to fall back on — the lexicon is the only signal."""
        from src.bracket_eval.pipeline import replay_conversation

        result = replay_conversation(
            [
                "biwi ne bola jaldi aana",
                "shaadi ki saalgirah hai",
                "landlord ko rent bhi dena hai",
            ],
            true_band="adult",
        )
        assert result.predicted_band == "adult", (
            "Hindi-dominant adult text must not abstain — upstream abstains here"
        )
