"""Tests for the CoNLL-2003 data, prompt, parsing and scoring code (no model, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conll

# ---------------------------------------------------------------------------
# IOB decoding
# ---------------------------------------------------------------------------


def test_iob2_spans() -> None:
    tags = ["B-PER", "I-PER", "O", "B-LOC", "B-LOC", "I-LOC", "O"]
    assert conll.iob_spans(tags) == [("PER", 0, 2), ("LOC", 3, 4), ("LOC", 4, 6)]


def test_iob1_spans_follow_conlleval() -> None:
    # IOB1: I- starts an entity after O or another type; B- only separates same-type neighbours.
    tags = ["I-ORG", "I-ORG", "I-PER", "O", "I-LOC", "B-LOC"]
    assert conll.iob_spans(tags) == [("ORG", 0, 2), ("PER", 2, 3), ("LOC", 4, 5), ("LOC", 5, 6)]


def test_entity_at_end_and_empty() -> None:
    assert conll.iob_spans(["O", "B-MISC", "I-MISC"]) == [("MISC", 1, 3)]
    assert conll.iob_spans([]) == []
    assert conll.iob_spans(["O", "O"]) == []


@pytest.mark.parametrize("bad", ["PER", "X-PER", "B-", "b-PER"])
def test_iob_rejects_malformed_tags(bad: str) -> None:
    with pytest.raises(ValueError, match="IOB"):
        conll.iob_spans(["O", bad])


def test_entities_from_tags_joins_tokens_and_keeps_repeats() -> None:
    tokens = ["Peter", "Blackburn", "met", "Peter", "Blackburn", "in", "New", "York"]
    tags = ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
    gold = conll.entities_from_tags(tokens, tags)
    assert gold == {
        "PER": ["Peter Blackburn", "Peter Blackburn"],
        "ORG": [],
        "LOC": ["New York"],
        "MISC": [],
    }


def test_entities_from_tags_validates() -> None:
    with pytest.raises(ValueError, match="tokens"):
        conll.entities_from_tags(["a"], ["O", "O"])
    with pytest.raises(ValueError, match="unknown entity type"):
        conll.entities_from_tags(["a"], ["B-DATE"])


def test_sentence_from_row_with_label_ids() -> None:
    row = {"tokens": ["EU", "rejects", "German", "call"], "ner_tags": [3, 0, 7, 0]}
    s = conll.sentence_from_row(row, "test", 5)
    assert s.id == "test-5" and s.source_split == "test"
    assert s.text == "EU rejects German call"
    assert s.gold["ORG"] == ["EU"] and s.gold["MISC"] == ["German"]


def test_tag_names_match_dataset_order() -> None:
    assert conll.NER_TAG_NAMES[0] == "O" and len(conll.NER_TAG_NAMES) == 9
    assert {t.split("-")[1] for t in conll.NER_TAG_NAMES[1:]} == set(conll.ENTITY_TYPES)


# ---------------------------------------------------------------------------
# Pool sampling
# ---------------------------------------------------------------------------


def _pool(n: int) -> list:
    return [
        conll.Sentence(f"test-{i}", "test", ("w",), {t: [] for t in conll.ENTITY_TYPES})
        for i in range(n)
    ]


def test_sample_pool_is_deterministic_and_ordered() -> None:
    pool = _pool(50)
    a = conll.sample_pool(pool, 10, seed=3)
    b = conll.sample_pool(pool, 10, seed=3)
    assert [s.id for s in a] == [s.id for s in b]
    idx = [int(s.id.split("-")[1]) for s in a]
    assert idx == sorted(idx) and len(set(idx)) == 10
    assert [s.id for s in conll.sample_pool(pool, 10, seed=4)] != [s.id for s in a]


def test_sample_pool_whole_and_invalid() -> None:
    pool = _pool(5)
    assert conll.sample_pool(pool, None) == pool
    assert conll.sample_pool(pool, 99) == pool
    with pytest.raises(ValueError, match="positive"):
        conll.sample_pool(pool, 0)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_prompt_lists_every_field_and_the_sentence() -> None:
    p = conll.build_prompt("Japan beat Syria .")
    for t in conll.ENTITY_TYPES:
        assert t in p
    assert "Sentence: Japan beat Syria .\nOutput:" in p
    assert conll.build_messages("x") == [{"role": "user", "content": conll.build_prompt("x")}]


def test_prompt_hash_is_stable_and_text_has_no_long_dashes() -> None:
    assert len(conll.prompt_sha256()) == 64
    assert "\u2014" not in conll.PROMPT_TEMPLATE and "\u2013" not in conll.PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def test_parse_plain_json() -> None:
    ok, schema, ents = conll.parse_entities(
        '{"PER": ["John Smith"], "ORG": [], "LOC": ["Paris"], "MISC": []}'
    )
    assert ok and schema
    assert ents == {"PER": ["John Smith"], "ORG": [], "LOC": ["Paris"], "MISC": []}


def test_parse_code_fence_and_trailing_prose() -> None:
    text = 'Here you go:\n```json\n{"PER": [], "ORG": ["EU"], "LOC": [], "MISC": ["German"]}\n```\nDone.'
    ok, schema, ents = conll.parse_entities(text)
    assert ok and schema and ents["ORG"] == ["EU"] and ents["MISC"] == ["German"]
    ok, _, ents = conll.parse_entities('{"PER": ["A"]} and then {"PER": ["B"]}')
    assert ok and ents["PER"] == ["A"]


def test_parse_lenient_values_but_flags_schema() -> None:
    ok, schema, ents = conll.parse_entities(
        '{"per": "Ann", "Org": null, "LOC": ["  New   York "], "misc": [3, "Euro"]}'
    )
    assert ok and not schema
    assert ents == {"PER": ["Ann"], "ORG": [], "LOC": ["New York"], "MISC": ["Euro"]}
    ok, schema, ents = conll.parse_entities(
        '{"PER": [], "ORG": [], "LOC": [], "MISC": [], "DATE": ["1996"]}'
    )
    assert ok and not schema and all(v == [] for v in ents.values())


@pytest.mark.parametrize(
    "text", ["", "no json here", "{'PER': ['single quotes']}", '{"PER": ["x"],}', "[1, 2]"]
)
def test_parse_failures(text: str) -> None:
    ok, schema, ents = conll.parse_entities(text)
    assert not ok and not schema and all(v == [] for v in ents.values())


def test_parse_skips_a_broken_brace_before_a_good_object() -> None:
    ok, _, ents = conll.parse_entities('{oops} {"LOC": ["Rome"]}')
    assert ok and ents["LOC"] == ["Rome"]


def test_normalize_entity() -> None:
    assert conll.normalize_entity("  New \t York ") == "New York"
    assert conll.normalize_entity("\uff21BC") == "ABC"  # NFKC folds full-width letters
    assert conll.normalize_entity("JAPAN") != conll.normalize_entity("Japan")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_exact_match_and_counts() -> None:
    gold = {"PER": ["Guy Hellers"], "ORG": [], "LOC": [], "MISC": ["Belgian", "Swiss"]}
    s = conll.score_entities({"PER": ["Guy Hellers"], "MISC": ["Swiss", "Belgian"]}, gold)
    assert (s.exact_match, s.tp, s.fp, s.fn) == (1, 3, 0, 0)
    s = conll.score_entities({"ORG": ["Belgian"], "MISC": ["Guy Hellers"]}, gold)
    assert (s.exact_match, s.tp, s.fp, s.fn) == (0, 0, 2, 3)
    assert s.per_type == {"PER": [0, 0, 1], "ORG": [0, 1, 0], "LOC": [0, 0, 0], "MISC": [0, 1, 2]}


def test_score_uses_sets_of_normalized_strings() -> None:
    gold = {"PER": ["Bernardin", "Bernardin"]}
    s = conll.score_entities({"PER": ["Bernardin"]}, gold)
    assert (s.exact_match, s.tp, s.fp, s.fn) == (1, 1, 0, 0)
    s = conll.score_entities({"PER": ["Bernardin ", " Bernardin"]}, gold)
    assert (s.tp, s.fp) == (1, 0)


def test_score_type_matters_and_case_matters() -> None:
    gold = {"LOC": ["JAPAN"]}
    assert conll.score_entities({"ORG": ["JAPAN"]}, gold).tp == 0
    assert conll.score_entities({"LOC": ["Japan"]}, gold).tp == 0


def test_parse_failure_is_never_an_exact_match() -> None:
    empty = {t: [] for t in conll.ENTITY_TYPES}
    assert conll.score_entities(empty, empty, parse_ok=True).exact_match == 1
    assert conll.score_entities(empty, empty, parse_ok=False).exact_match == 0


def test_score_ignores_empty_strings() -> None:
    s = conll.score_entities({"PER": ["", "  "]}, {"PER": []})
    assert (s.exact_match, s.fp) == (1, 0)
