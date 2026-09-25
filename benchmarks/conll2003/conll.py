"""CoNLL-2003 data, prompt, output parsing and scoring for the UCCI replication.

This module holds everything about the task that does not need a model:

* loading the public CoNLL-2003 English NER data (Tjong Kim Sang and De Meulder,
  2003) from a pinned Hugging Face revision and turning its IOB tags into
  entity strings,
* the JSON-extraction prompt, written in the spirit of the prompt in the
  paper's Appendix B.1 (a short instruction plus the list of JSON fields),
* parsing a model's raw output back into entity lists, and
* scoring a prediction against the gold entities: the exact-match event used
  as e(x) for calibration (paper Section 4.2) and entity-level true
  positive, false positive and false negative counts for micro-F1 (the
  evaluation metric of paper Section 6).

Only the standard library is imported at module load, so the parsing and
scoring logic can be tested without torch or ``datasets``.

Scoring convention
------------------
The model returns entity *strings*, not token offsets, so each sentence is
scored on the set of ``(type, normalized string)`` pairs. A mention that
occurs twice in a sentence counts once. Normalization applies Unicode NFKC,
trims the ends and collapses internal whitespace; it is case-sensitive, so
``"JAPAN"`` and ``"Japan"`` are different strings, exactly as they appear in
the sentence. An output that does not parse as a JSON object scores as an
empty prediction and is never an exact match.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DATASET_ID",
    "DATASET_REVISION",
    "DATASET_REVISION_REF",
    "ENTITY_TYPES",
    "NER_TAG_NAMES",
    "PROMPT_TEMPLATE",
    "PROMPT_VERSION",
    "EntityScore",
    "Sentence",
    "build_messages",
    "build_prompt",
    "entities_from_tags",
    "extract_json_object",
    "iob_spans",
    "load_pool",
    "normalize_entity",
    "parse_entities",
    "prompt_sha256",
    "sample_pool",
    "score_entities",
    "sentence_from_row",
]

#: Hugging Face dataset id of the canonical CoNLL-2003 loader.
DATASET_ID = "eriktks/conll2003"
#: Commit of the dataset's automatic parquet conversion. The original
#: loading script does not run under ``datasets`` 4.x, the parquet branch
#: does. Pinning the commit (not the moving branch name) fixes the bytes.
DATASET_REVISION = "ce85b39f9dd99f552d0739d456814e95fb6a39b0"
#: The branch the pinned commit was read from, for the record.
DATASET_REVISION_REF = "refs/convert/parquet"

#: Entity types of CoNLL-2003, in the order used in the prompt and the logs.
ENTITY_TYPES: Tuple[str, ...] = ("PER", "ORG", "LOC", "MISC")

#: ``ner_tags`` label names of the Hugging Face CoNLL-2003 dataset.
NER_TAG_NAMES: Tuple[str, ...] = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
)

#: Bump when the prompt text changes; logged with every run.
PROMPT_VERSION = "conll2003-json-v1"

#: The user message sent to every model (both models see identical prompts,
#: as in paper Section 6.1). ``{sentence}`` is the space-joined CoNLL tokens.
#: The wording (type glosses, "only proper names") was settled on sentences
#: from the CoNLL-2003 train split, which is never part of the evaluation pool.
PROMPT_TEMPLATE = (
    "Extract the named entities from this sentence.\n"
    "Return only a JSON object with fields: PER, ORG, LOC, MISC.\n"
    "Each field is a list of names copied exactly from the sentence; "
    "use an empty list when there are none.\n"
    "PER: people. ORG: organizations, companies, sports teams, political parties. "
    "LOC: countries, cities, regions and other places. "
    "MISC: other proper names, such as nationalities, languages, events and titles.\n"
    "Only include proper names. Do not include numbers, dates, scores or common nouns.\n"
    "\n"
    "Sentence: {sentence}\n"
    "Output:"
)


@dataclass(frozen=True)
class Sentence:
    """One CoNLL-2003 sentence with its gold entities.

    Attributes
    ----------
    id : str
        Stable identifier ``"<source_split>-<row index>"``, for example
        ``"test-17"``. Unique across the pooled splits.
    source_split : str
        The CoNLL-2003 split the sentence comes from (``"train"``,
        ``"validation"`` or ``"test"``). This is not the UCCI
        calibration/validation/test split, which is drawn later from the pool.
    tokens : tuple of str
        The CoNLL tokens.
    gold : dict
        Entity type to the list of entity strings in order of appearance
        (repeated mentions kept).
    """

    id: str
    source_split: str
    tokens: Tuple[str, ...]
    gold: Dict[str, List[str]] = field(hash=False, compare=False)

    @property
    def text(self) -> str:
        """The sentence as shown to the model: tokens joined by single spaces."""
        return " ".join(self.tokens)


@dataclass(frozen=True)
class EntityScore:
    """Scoring outcome of one prediction.

    Attributes
    ----------
    exact_match : int
        1 when the output parsed and its entity set equals the gold set for
        every type, else 0. The calibration error event is ``e = 1 - exact_match``
        (paper Section 4.2).
    tp, fp, fn : int
        Entity-level counts summed over types, for micro-F1.
    per_type : dict
        Type to ``[tp, fp, fn]``.
    """

    exact_match: int
    tp: int
    fp: int
    fn: int
    per_type: Dict[str, List[int]]


def iob_spans(tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Decode IOB tags into ``(type, start, end)`` spans with ``end`` exclusive.

    Follows the chunking rule of the CoNLL ``conlleval`` script, so it reads
    both IOB2 (every entity starts with ``B-``) and IOB1 (``B-`` only between
    adjacent entities of the same type): a chunk starts at ``B-X``, or at
    ``I-X`` when the previous tag is ``O`` or of another type.

    Parameters
    ----------
    tags : sequence of str
        Tags such as ``"O"``, ``"B-PER"``, ``"I-PER"``.

    Returns
    -------
    list of (str, int, int)

    Raises
    ------
    ValueError
        If a tag is neither ``"O"`` nor of the form ``"B-X"`` / ``"I-X"``.
    """
    spans: List[Tuple[str, int, int]] = []
    cur_type: Optional[str] = None
    start = 0
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur_type is not None:
                spans.append((cur_type, start, i))
                cur_type = None
            continue
        prefix, sep, typ = tag.partition("-")
        if sep != "-" or prefix not in ("B", "I") or not typ:
            raise ValueError(f"tag {i} is not an IOB tag: {tag!r}")
        if prefix == "B" or cur_type != typ:
            if cur_type is not None:
                spans.append((cur_type, start, i))
            cur_type, start = typ, i
    if cur_type is not None:
        spans.append((cur_type, start, len(tags)))
    return spans


def entities_from_tags(tokens: Sequence[str], tags: Sequence[str]) -> Dict[str, List[str]]:
    """Gold entity strings per type from tokens and IOB tags.

    Parameters
    ----------
    tokens : sequence of str
    tags : sequence of str
        Same length as ``tokens``.

    Returns
    -------
    dict
        Every type in :data:`ENTITY_TYPES` mapped to its entity strings in
        order of appearance (tokens joined by single spaces).

    Raises
    ------
    ValueError
        On a length mismatch or an entity type outside :data:`ENTITY_TYPES`.
    """
    if len(tokens) != len(tags):
        raise ValueError(f"{len(tokens)} tokens but {len(tags)} tags")
    out: Dict[str, List[str]] = {t: [] for t in ENTITY_TYPES}
    for typ, a, b in iob_spans(tags):
        if typ not in out:
            raise ValueError(f"unknown entity type {typ!r}; expected one of {ENTITY_TYPES}")
        out[typ].append(" ".join(tokens[a:b]))
    return out


def sentence_from_row(row: Mapping[str, Any], source_split: str, index: int) -> Sentence:
    """Build a :class:`Sentence` from one Hugging Face CoNLL-2003 row.

    Parameters
    ----------
    row : mapping
        Needs ``"tokens"`` (list of str) and ``"ner_tags"`` (list of int
        label ids in :data:`NER_TAG_NAMES` order, or list of str tags).
    source_split : str
    index : int
        Row index inside ``source_split``, used for the stable id.
    """
    tokens = tuple(str(t) for t in row["tokens"])
    raw_tags = list(row["ner_tags"])
    tags = [NER_TAG_NAMES[int(t)] if not isinstance(t, str) else t for t in raw_tags]
    return Sentence(
        id=f"{source_split}-{index}",
        source_split=source_split,
        tokens=tokens,
        gold=entities_from_tags(tokens, tags),
    )


def load_pool(
    source_splits: Iterable[str] = ("validation", "test"),
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
) -> List[Sentence]:
    """Load and pool CoNLL-2003 sentences from the pinned dataset revision.

    Sentences are returned in dataset order, split by split. Rows with no
    tokens and ``-DOCSTART-`` markers are skipped (the pinned revision
    contains neither, the guard is for other copies of the data).

    Parameters
    ----------
    source_splits : iterable of str
        CoNLL-2003 splits to pool. The replication uses the validation and
        test splits (6,703 sentences); the train split is never needed
        because UCCI trains nothing but the calibration map.
    dataset_id, revision : str
        Hugging Face dataset id and commit.

    Returns
    -------
    list of Sentence

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    """
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "loading CoNLL-2003 needs the 'datasets' package: pip install 'datasets>=2.14'"
        ) from exc
    pool: List[Sentence] = []
    for split in source_splits:
        ds = datasets.load_dataset(dataset_id, revision=revision, split=split)
        names = ds.features["ner_tags"].feature.names
        if tuple(names) != NER_TAG_NAMES:
            raise ValueError(f"unexpected ner_tags label names {names}; expected {NER_TAG_NAMES}")
        for i, row in enumerate(ds):
            if not row["tokens"] or row["tokens"][0] == "-DOCSTART-":
                continue
            pool.append(sentence_from_row(row, split, i))
    return pool


def sample_pool(pool: Sequence[Sentence], limit: Optional[int], seed: int = 0) -> List[Sentence]:
    """Deterministic subset of ``pool`` for quick runs.

    Returns the whole pool when ``limit`` is None or not smaller than the
    pool, else the first ``limit`` sentences of a seeded permutation, put
    back in pool order. Two runs with the same pool, ``limit`` and ``seed``
    (for example the small and the large model) get the same sentences.
    """
    if limit is None or limit >= len(pool):
        return list(pool)
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    import random

    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    keep = sorted(order[:limit])
    return [pool[i] for i in keep]


def build_prompt(sentence_text: str) -> str:
    """The user message for one sentence (see :data:`PROMPT_TEMPLATE`)."""
    return PROMPT_TEMPLATE.format(sentence=sentence_text)


def build_messages(sentence_text: str) -> List[Dict[str, str]]:
    """Chat messages for an instruction-tuned model: one user turn.

    No system message is added, so a model's chat template applies its own
    default (Qwen2.5 inserts its standard system prompt).
    """
    return [{"role": "user", "content": build_prompt(sentence_text)}]


def prompt_sha256() -> str:
    """SHA-256 of the prompt template, logged to detect prompt drift on resume."""
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """First JSON object in a model output, or None.

    Looks inside a Markdown code fence first (instruction-tuned models often
    add one), then in the raw text. Parsing starts at the first ``{`` and
    stops at the end of the first complete JSON value, so trailing prose is
    ignored. Nothing else is repaired: single quotes, comments or trailing
    commas make the output unparseable.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    decoder = json.JSONDecoder()
    for cand in candidates:
        start = cand.find("{")
        while start >= 0:
            try:
                obj, _ = decoder.raw_decode(cand[start:])
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                return obj
            start = cand.find("{", start + 1)
    return None


def normalize_entity(s: str) -> str:
    """NFKC, trimmed, internal whitespace collapsed; case is kept."""
    return " ".join(unicodedata.normalize("NFKC", s).split())


def _as_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def parse_entities(text: str) -> Tuple[bool, bool, Dict[str, List[str]]]:
    """Parse a raw model output into entity lists.

    Parameters
    ----------
    text : str
        Decoded model output.

    Returns
    -------
    parse_ok : bool
        True when a JSON object was found.
    schema_ok : bool
        True when the object has exactly the four fields (keys compared
        case-insensitively) and every value is a list of strings.
    entities : dict
        Every type in :data:`ENTITY_TYPES` mapped to the predicted strings
        (normalized, empty strings dropped, order kept). Keys are matched
        case-insensitively, a bare string value counts as a one-item list,
        ``null`` as empty, and non-string list items are dropped. Other keys
        are ignored.
    """
    empty: Dict[str, List[str]] = {t: [] for t in ENTITY_TYPES}
    obj = extract_json_object(text)
    if obj is None:
        return False, False, empty
    by_key = {str(k).strip().upper(): v for k, v in obj.items()}
    schema_ok = set(by_key) == set(ENTITY_TYPES) and all(
        isinstance(v, list) and all(isinstance(x, str) for x in v) for v in by_key.values()
    )
    out: Dict[str, List[str]] = {}
    for t in ENTITY_TYPES:
        vals = [normalize_entity(s) for s in _as_strings(by_key.get(t))]
        out[t] = [v for v in vals if v]
    return True, schema_ok, out


def score_entities(
    pred: Mapping[str, Sequence[str]],
    gold: Mapping[str, Sequence[str]],
    parse_ok: bool = True,
) -> EntityScore:
    """Score predicted entities against gold entities for one sentence.

    Both sides are reduced to sets of normalized strings per type (see the
    module docstring). ``tp`` counts predicted strings that are gold,
    ``fp`` predicted strings that are not, ``fn`` gold strings not predicted.

    Parameters
    ----------
    pred, gold : mapping
        Type to entity strings. Missing types count as empty.
    parse_ok : bool
        False forces ``exact_match = 0`` even when both sides are empty, so
        an unparseable answer is always an error event for calibration.

    Returns
    -------
    EntityScore
    """
    tp = fp = fn = 0
    per_type: Dict[str, List[int]] = {}
    all_equal = True
    for t in ENTITY_TYPES:
        p = {normalize_entity(s) for s in pred.get(t, ())} - {""}
        g = {normalize_entity(s) for s in gold.get(t, ())} - {""}
        a, b, c = len(p & g), len(p - g), len(g - p)
        per_type[t] = [a, b, c]
        tp, fp, fn = tp + a, fp + b, fn + c
        all_equal = all_equal and p == g
    return EntityScore(int(parse_ok and all_equal), tp, fp, fn, per_type)
