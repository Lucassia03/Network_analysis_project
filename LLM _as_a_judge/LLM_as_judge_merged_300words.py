"""
Comment-specific conversational context extraction for Reddit-style threads.

This script reconstructs Reddit-style post/comment threads and extracts the
specific context each target comment appears to be responding to.

It intentionally does NOT perform stance detection, sentiment analysis,
toxicity classification, agreement/disagreement classification, support/
opposition labeling, or political-position classification.

Supported input formats:
- pandas DataFrame
- list[dict]
- dict-of-lists
- single dict

Output is returned as Python data structures, not JSON text.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


Record = Dict[str, Any]
RecordsLike = Union[Sequence[Mapping[str, Any]], Mapping[str, Any], Any]


POST_ID_ALIASES = ("post_id", "submission_id", "link_id", "thread_id", "id")
COMMENT_ID_ALIASES = ("comment_id", "id", "commentid", "name")
PARENT_ID_ALIASES = ("parent_comment_id", "parent_id", "parent", "reply_to", "in_reply_to")
POST_TITLE_ALIASES = ("title", "post_title", "submission_title")
POST_BODY_ALIASES = ("body", "selftext", "post_body", "text", "content")
COMMENT_TEXT_ALIASES = ("body", "text", "comment_text", "comment_body", "content")
AUTHOR_ALIASES = ("author", "user", "username")
CREATED_ALIASES = ("created_utc", "created", "timestamp", "created_at")
ORDER_ALIASES = ("timestamp_order", "order", "created_order", "sequence", "seq")
DEPTH_ALIASES = ("depth", "level")
SCORE_ALIASES = ("score", "ups", "upvotes")


CONTEXT_TYPES = {
    "post_direct",
    "parent_reply",
    "ancestor_chain",
    "prior_sibling_reference",
    "broader_thread_reference",
    "topic_shift",
    "ambiguous",
}


LLM_JUDGE_REQUIRED_FIELDS = {
    "context_type",
    "target_context",
    "resolved_references",
    "confidence",
    "ambiguities",
}

LLM_REFERENCE_SOURCES = {
    "direct_parent",
    "ancestor_comment",
    "prior_sibling",
    "prior_thread_comment",
    "post_title",
    "post_body",
    "post",
    "unknown",
}

LLM_JUDGE_BANNED_LABEL_RE = re.compile(
    r"\b("
    r"stance|sentiment|toxicity|toxic|"
    r"agree|agrees|agreed|agreement|disagree|disagrees|disagreed|disagreement|"
    r"support|supports|supported|supporting|"
    r"oppose|opposes|opposed|opposing|opposition|"
    r"political\s+position|ideology|ideological|morality|moral|correctness|"
    r"pro|anti"
    r")\b|\bpro-|\banti-",
    re.I,
)

LOW_QUALITY_CONTEXT_RE = re.compile(
    r"^\s*(unknown|unclear|not\s+sure|n/?a|none|null|"
    r"the\s+context|prior\s+context|previous\s+context|conversation)\s*$",
    re.I,
)


MIN_TARGET_CONTEXT_WORDS = 300
POST_CONTEXT_PRIORITY_BOOST = 0.14
POST_CONTEXT_NESTED_TITLE_BASE = 0.66
POST_CONTEXT_NESTED_BODY_BASE = 0.62
POST_CONTEXT_TOP_LEVEL_TITLE_BASE = 0.94
POST_CONTEXT_TOP_LEVEL_BODY_BASE = 0.90


REFERENCE_PATTERNS = [
    ("those people", re.compile(r"\bthose\s+people\b", re.I)),
    ("what do you mean", re.compile(r"\bwhat\s+do\s+you\s+mean\b|\bwym\b", re.I)),
    ("the parent comment", re.compile(r"\bthe\s+parent\s+comment\b", re.I)),
    ("the article", re.compile(r"\bthe\s+article\b", re.I)),
    ("the study", re.compile(r"\bthe\s+study\b", re.I)),
    ("the video", re.compile(r"\bthe\s+video\b", re.I)),
    ("the title", re.compile(r"\bthe\s+title\b", re.I)),
    ("this", re.compile(r"\bthis\b", re.I)),
    ("that", re.compile(r"\bthat\b", re.I)),
    ("it", re.compile(r"\bit\b", re.I)),
    ("they", re.compile(r"\bthey\b", re.I)),
    ("them", re.compile(r"\bthem\b", re.I)),
    ("he", re.compile(r"\bhe\b", re.I)),
    ("she", re.compile(r"\bshe\b", re.I)),
    ("same", re.compile(r"\bsame\b", re.I)),
    ("exactly", re.compile(r"\bexactly\b", re.I)),
    ("yes", re.compile(r"^\s*yes\b|\byep\b|\byeah\b", re.I)),
    ("no", re.compile(r"^\s*no\b|\bnope\b|\bnah\b", re.I)),
    ("agreed", re.compile(r"\bagreed\b|\bi\s+agree\b", re.I)),
    ("why", re.compile(r"^\s*why\b|\bwhy\??\s*$", re.I)),
    ("source", re.compile(r"\bsource\b|\bcitation\b|\blink\??\s*$", re.I)),
    ("OP", re.compile(r"\bOP\b", re.I)),
]


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
QUOTE_LINE_RE = re.compile(r"(?m)^\s*>+\s?(.*)$")
SARCASM_MARKER_RE = re.compile(
    r"\b/s\b|yeah\s+right|sure\s+jan|totally\s+(?:not|fine)|as\s+if",
    re.I,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "hers", "him", "his", "i", "if", "in", "into",
    "is", "it", "its", "just", "me", "my", "no", "not", "of", "on", "or", "our",
    "ours", "she", "so", "some", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your", "yours", "about", "after", "all", "also", "am",
    "any", "being", "more", "most", "one", "only", "out", "over", "same",
    "should", "than", "through", "under", "up", "very",
}


@dataclass
class ThreadData:
    """Normalized thread data and lookup indexes."""

    posts: List[Record]
    comments: List[Record]
    post_index: Dict[str, Record]
    comment_index: Dict[str, Record]
    comments_by_post: Dict[str, List[Record]]
    children_by_parent: Dict[Tuple[str, str], List[Record]]
    post_id_col: Optional[str] = None
    comment_id_col: Optional[str] = None
    parent_id_col: Optional[str] = None


def normalize_text(text: Any) -> str:
    """Normalize text while preserving readable content."""
    if text is None:
        return ""

    try:
        if isinstance(text, float) and math.isnan(text):
            return ""
    except Exception:
        pass

    value = str(text)
    if value.strip().lower() in {"nan", "none", "null"}:
        return ""

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = URL_RE.sub("[URL]", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def is_deleted_or_empty(text: Any) -> bool:
    """Return True for deleted, removed, null, or empty comment text."""
    value = normalize_text(text).strip().lower()
    return not value or value in {"[deleted]", "[removed]", "deleted", "removed", "comment deleted"}


def to_records(data: RecordsLike) -> List[Record]:
    """Convert pandas DataFrames or dict/list inputs into list[dict]."""
    if data is None:
        return []

    if hasattr(data, "to_dict"):
        try:
            return [dict(row) for row in data.to_dict(orient="records")]
        except TypeError:
            pass

    if isinstance(data, Mapping):
        if all(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in data.values()):
            keys = list(data.keys())
            lengths = [len(data[key]) for key in keys]
            if len(set(lengths)) > 1:
                raise ValueError("Dictionary-of-lists input has unequal column lengths.")
            return [{key: data[key][i] for key in keys} for i in range(lengths[0] if lengths else 0)]
        return [dict(data)]

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [dict(row) for row in data]

    raise TypeError("Input must be a pandas DataFrame, mapping, or sequence of mappings.")


def _lower_column_map(records: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Map lowercase column names to original column names."""
    columns = set()
    for record in records:
        columns.update(str(key) for key in record.keys())
    return {column.lower(): column for column in columns}


def detect_column(records: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> Optional[str]:
    """Find the first existing alias in records."""
    lower_map = _lower_column_map(records)
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def detect_text_column(df_or_records: RecordsLike, kind: str = "comment") -> Optional[str]:
    """Detect the likely text column for posts or comments."""
    records = to_records(df_or_records)
    aliases = POST_BODY_ALIASES if kind == "post" else COMMENT_TEXT_ALIASES
    return detect_column(records, aliases)


def detect_id_columns(posts: RecordsLike, comments: RecordsLike) -> Dict[str, Optional[str]]:
    """Detect post, comment, and parent identifier columns."""
    post_records = to_records(posts)
    comment_records = to_records(comments)

    return {
        "post_id": detect_column(post_records, POST_ID_ALIASES),
        "comment_post_id": detect_column(comment_records, ("post_id", "submission_id", "link_id", "thread_id")),
        "comment_id": detect_column(comment_records, COMMENT_ID_ALIASES),
        "parent_id": detect_column(comment_records, PARENT_ID_ALIASES),
    }


def canonical_id(value: Any) -> Optional[str]:
    """Canonicalize Reddit-like IDs, including t1_/t3_ prefixes and integral floats."""
    if value is None:
        return None

    try:
        if isinstance(value, float):
            if math.isnan(value):
                return None
            if value.is_integer():
                return str(int(value))
    except Exception:
        pass

    value_str = str(value).strip()
    if not value_str or value_str.lower() in {"nan", "none", "null"}:
        return None

    if re.fullmatch(r"\d+\.0+", value_str):
        return value_str.split(".", 1)[0]

    if value_str.startswith(("t1_", "t3_")):
        return value_str[3:]
    return value_str



def _safe_get(record: Mapping[str, Any], column: Optional[str], default: Any = None) -> Any:
    """Safely read a detected column."""
    if column is None:
        return default
    return record.get(column, default)


def _numeric_or_none(value: Any) -> Optional[float]:
    """Convert value to float when possible."""
    if value is None:
        return None

    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return None


def _record_order_key(record: Mapping[str, Any]) -> Tuple[int, float, int]:
    """
    Chronological sort key.

    Preference:
    1. timestamp_order
    2. created_utc
    3. input row order
    """
    timestamp_order = _numeric_or_none(record.get("_timestamp_order"))
    if timestamp_order is not None:
        return (0, timestamp_order, int(record.get("_row_order", 0)))

    created_utc = _numeric_or_none(record.get("_created_utc"))
    if created_utc is not None:
        return (1, created_utc, int(record.get("_row_order", 0)))

    row_order = int(record.get("_row_order", 0))
    return (2, float(row_order), row_order)


def _order_before(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return True when left appears chronologically before right."""
    return _record_order_key(left) < _record_order_key(right)


def normalize_posts(posts: RecordsLike) -> List[Record]:
    """Normalize post records into internal canonical fields."""
    records = to_records(posts)

    post_id_col = detect_column(records, POST_ID_ALIASES)
    title_col = detect_column(records, POST_TITLE_ALIASES)
    body_col = detect_column(records, POST_BODY_ALIASES)
    author_col = detect_column(records, AUTHOR_ALIASES)
    created_col = detect_column(records, CREATED_ALIASES)

    normalized: List[Record] = []
    for index, record in enumerate(records):
        raw_post_id = _safe_get(record, post_id_col, index)
        post_id = canonical_id(raw_post_id) or str(index)

        output = dict(record)
        output["_post_id"] = post_id
        output["_raw_post_id"] = raw_post_id
        output["_title"] = normalize_text(_safe_get(record, title_col, ""))
        output["_body"] = normalize_text(_safe_get(record, body_col, ""))
        output["_author"] = _safe_get(record, author_col)
        output["_created_utc"] = _safe_get(record, created_col)
        output["_row_order"] = index
        normalized.append(output)

    return normalized


def normalize_comments(comments: RecordsLike) -> List[Record]:
    """Normalize comment records into internal canonical fields."""
    records = to_records(comments)

    comment_id_col = detect_column(records, COMMENT_ID_ALIASES)
    post_id_col = detect_column(records, ("post_id", "submission_id", "link_id", "thread_id"))
    parent_id_col = detect_column(records, PARENT_ID_ALIASES)
    text_col = detect_column(records, COMMENT_TEXT_ALIASES)
    author_col = detect_column(records, AUTHOR_ALIASES)
    created_col = detect_column(records, CREATED_ALIASES)
    order_col = detect_column(records, ORDER_ALIASES)
    depth_col = detect_column(records, DEPTH_ALIASES)
    score_col = detect_column(records, SCORE_ALIASES)

    normalized: List[Record] = []
    for index, record in enumerate(records):
        raw_comment_id = _safe_get(record, comment_id_col, index)
        comment_id = canonical_id(raw_comment_id) or str(index)

        output = dict(record)
        output["_comment_id"] = comment_id
        output["_raw_comment_id"] = raw_comment_id
        output["_post_id"] = canonical_id(_safe_get(record, post_id_col))
        output["_raw_post_id"] = _safe_get(record, post_id_col)
        output["_parent_id"] = canonical_id(_safe_get(record, parent_id_col))
        output["_raw_parent_id"] = _safe_get(record, parent_id_col)
        output["_body"] = normalize_text(_safe_get(record, text_col, ""))
        output["_author"] = _safe_get(record, author_col)
        output["_created_utc"] = _safe_get(record, created_col)
        output["_timestamp_order"] = _safe_get(record, order_col)
        output["_depth"] = _safe_get(record, depth_col)
        output["_score"] = _safe_get(record, score_col)
        output["_row_order"] = index
        normalized.append(output)

    return normalized


def build_post_index(posts: RecordsLike) -> Dict[str, Record]:
    """Build post_id -> post record."""
    return {post["_post_id"]: post for post in normalize_posts(posts)}


def build_comment_index(comments: RecordsLike) -> Dict[str, Record]:
    """Build comment_id -> comment record."""
    return {comment["_comment_id"]: comment for comment in normalize_comments(comments)}


def prepare_thread_data(posts: RecordsLike, comments: RecordsLike) -> ThreadData:
    """Normalize input data and build traversal indexes."""
    raw_posts = to_records(posts)
    raw_comments = to_records(comments)
    detected = detect_id_columns(raw_posts, raw_comments)

    normalized_posts = normalize_posts(raw_posts)
    normalized_comments = normalize_comments(raw_comments)

    post_index = {post["_post_id"]: post for post in normalized_posts}
    comment_index = {comment["_comment_id"]: comment for comment in normalized_comments}

    comments_by_post: Dict[str, List[Record]] = defaultdict(list)
    children_by_parent: Dict[Tuple[str, str], List[Record]] = defaultdict(list)

    for comment in normalized_comments:
        post_id = comment.get("_post_id")
        if post_id is None:
            continue

        comments_by_post[post_id].append(comment)

        parent_id = comment.get("_parent_id")
        parent_key = post_id if parent_id is None or parent_id == post_id else parent_id
        children_by_parent[(post_id, parent_key)].append(comment)

    for grouped_comments in comments_by_post.values():
        grouped_comments.sort(key=_record_order_key)

    for grouped_children in children_by_parent.values():
        grouped_children.sort(key=_record_order_key)

    return ThreadData(
        posts=normalized_posts,
        comments=normalized_comments,
        post_index=post_index,
        comment_index=comment_index,
        comments_by_post=dict(comments_by_post),
        children_by_parent=dict(children_by_parent),
        post_id_col=detected.get("post_id"),
        comment_id_col=detected.get("comment_id"),
        parent_id_col=detected.get("parent_id"),
    )


def get_comment_order(comments: RecordsLike) -> Dict[str, int]:
    """Return comment_id -> chronological rank."""
    ordered = sorted(normalize_comments(comments), key=_record_order_key)
    return {comment["_comment_id"]: index for index, comment in enumerate(ordered)}


def get_direct_parent(comment: Mapping[str, Any], comment_index: Mapping[str, Record]) -> Optional[Record]:
    """Return the direct parent comment, if available."""
    parent_id = comment.get("_parent_id")
    post_id = comment.get("_post_id")

    if parent_id is None or parent_id == post_id:
        return None

    return comment_index.get(parent_id)


def get_ancestor_chain(comment: Mapping[str, Any], comment_index: Mapping[str, Record]) -> List[Record]:
    """Return ancestors from top-level ancestor down to direct parent."""
    ancestors_reversed: List[Record] = []
    seen = {comment.get("_comment_id")}
    current = comment

    while True:
        parent = get_direct_parent(current, comment_index)
        if parent is None:
            break

        parent_id = parent.get("_comment_id")
        if parent_id in seen:
            break

        seen.add(parent_id)
        ancestors_reversed.append(parent)
        current = parent

    return list(reversed(ancestors_reversed))


def get_prior_siblings(comment: Mapping[str, Any], comments: Sequence[Mapping[str, Any]]) -> List[Record]:
    """Return earlier comments with the same parent in the same post."""
    post_id = comment.get("_post_id")
    parent_id = comment.get("_parent_id") or post_id

    siblings = []
    for candidate in comments:
        if candidate.get("_comment_id") == comment.get("_comment_id"):
            continue
        if candidate.get("_post_id") != post_id:
            continue

        candidate_parent = candidate.get("_parent_id") or post_id
        if candidate_parent == parent_id and _order_before(candidate, comment):
            siblings.append(dict(candidate))

    siblings.sort(key=_record_order_key)
    return siblings


def get_prior_thread_comments(comment: Mapping[str, Any], comments: Sequence[Mapping[str, Any]]) -> List[Record]:
    """Return all earlier comments from the same post, excluding the target."""
    post_id = comment.get("_post_id")

    prior = [
        dict(candidate)
        for candidate in comments
        if candidate.get("_comment_id") != comment.get("_comment_id")
        and candidate.get("_post_id") == post_id
        and _order_before(candidate, comment)
    ]

    prior.sort(key=_record_order_key)
    return prior


def extract_keywords(text: str, max_terms: int = 12) -> List[str]:
    """Extract simple keyword-like tokens using transparent heuristics."""
    normalized = normalize_text(text).lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", normalized)
    filtered = [token.strip("'-") for token in tokens if token not in STOPWORDS and len(token) > 2]
    counts = Counter(filtered)

    for proper in re.findall(r"\b[A-Z][a-zA-Z0-9_'-]{2,}\b", normalize_text(text)):
        token = proper.lower()
        if token not in STOPWORDS:
            counts[token] += 1

    return [term for term, _ in counts.most_common(max_terms)]


def token_set(text: str) -> set:
    """Return a set of informative lowercase tokens."""
    return set(extract_keywords(text, max_terms=1000))


def lexical_overlap(left: str, right: str) -> float:
    """Compute normalized lexical overlap between two texts."""
    left_tokens = token_set(left)
    right_tokens = token_set(right)

    if not left_tokens or not right_tokens:
        return 0.0

    denominator = math.sqrt(len(left_tokens) * len(right_tokens))
    if denominator == 0:
        return 0.0

    return min(1.0, len(left_tokens & right_tokens) / denominator)


def excerpt(text: Any, max_chars: int = 240) -> str:
    """Return a compact, readable excerpt."""
    value = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if len(value) <= max_chars:
        return value

    cut = value[: max_chars - 1].rsplit(" ", 1)[0]
    return f"{cut}..."




def word_count(text: Any) -> int:
    """Count readable word tokens in a text string."""
    return len(re.findall(r"\b[\w'-]+\b", normalize_text(text)))


def _join_nonempty(parts: Iterable[str], separator: str = " ") -> str:
    """Join non-empty normalized text fragments."""
    return separator.join(part for part in (normalize_text(p) for p in parts) if part)


def _context_block(label: str, text_value: Any, max_chars: int = 1400) -> str:
    """Build a labeled context block from supplied source text."""
    value = excerpt(text_value, max_chars)
    if not value:
        return ""
    return f"{label}: {value}"


def build_expanded_target_context(
    target_context: str,
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    weighted_context: Sequence[Mapping[str, Any]],
    context_type: str,
    min_words: int = MIN_TARGET_CONTEXT_WORDS,
) -> str:
    """
    Expand the extracted conversational context to at least min_words.

    The expansion remains text-extraction focused: it uses only supplied prior
    post/comment text and transparent source labels. It does not add stance,
    sentiment, correctness, ideology, morality, or author-intent judgments.
    """
    base = normalize_text(target_context)
    if word_count(base) >= min_words:
        return base

    sections: List[str] = []
    if base:
        sections.append(
            "Extracted prior context. "
            f"The current context type is {context_type}. "
            f"The central prior text identified for the target comment is: {base}"
        )

    if post:
        title = normalize_text(post.get("_title", ""))
        body = normalize_text(post.get("_body", ""))
        post_parts: List[str] = []
        title_block = _context_block("Post title", title, max_chars=700)
        body_block = _context_block("Post body", body, max_chars=1800)
        if title_block:
            post_parts.append(title_block)
        if body_block:
            post_parts.append(body_block)
        if post_parts:
            sections.append(
                "Post-level context is given high priority because it defines the shared issue, entity, event, or question that all later comments can refer back to. "
                + " ".join(post_parts)
            )

    if parent:
        parent_block = _context_block("Direct parent comment", parent.get("_body", ""), max_chars=1500)
        if parent_block:
            sections.append(
                "Local reply context. The direct parent remains important because it is the closest prior text in the reply chain. "
                + parent_block
            )

    if ancestors:
        ancestor_blocks = []
        for index, ancestor in enumerate(ancestors[-4:], start=1):
            block = _context_block(
                f"Ancestor comment {index}",
                ancestor.get("_body", ""),
                max_chars=900,
            )
            if block:
                ancestor_blocks.append(block)
        if ancestor_blocks:
            sections.append(
                "Ancestor-chain context. These prior comments are included because they can contain the issue, claim, question, joke, or reference that the target comment inherits through the thread structure. "
                + " ".join(ancestor_blocks)
            )

    useful_siblings = [
        sibling for sibling in prior_siblings[-5:]
        if not is_deleted_or_empty(sibling.get("_body", ""))
    ]
    if useful_siblings:
        sibling_blocks = []
        for sibling in useful_siblings:
            block = _context_block(
                f"Earlier sibling comment {sibling.get('_comment_id')}",
                sibling.get("_body", ""),
                max_chars=700,
            )
            if block:
                sibling_blocks.append(block)
        if sibling_blocks:
            sections.append(
                "Earlier sibling context. These comments came before the target under the same parent and may supply a referent for short or vague replies. "
                + " ".join(sibling_blocks)
            )

    weighted_blocks = []
    for item in weighted_context:
        if item.get("source") == "target_comment":
            continue
        block = _context_block(
            f"Weighted candidate {item.get('source')} {item.get('id')} weight={item.get('relevance_weight')}",
            item.get("text_excerpt", ""),
            max_chars=500,
        )
        if block:
            weighted_blocks.append(block)
        if len(weighted_blocks) >= 8:
            break
    if weighted_blocks:
        sections.append(
            "Highest-weight prior context candidates. These are deterministic candidates selected before any model judgment and are used only as evidence for context extraction. "
            + " ".join(weighted_blocks)
        )

    broader_blocks = []
    local_ids = {target_comment.get("_comment_id")}
    if parent:
        local_ids.add(parent.get("_comment_id"))
    local_ids.update(ancestor.get("_comment_id") for ancestor in ancestors)
    local_ids.update(sibling.get("_comment_id") for sibling in prior_siblings)
    for comment in prior_thread_comments[-6:]:
        if comment.get("_comment_id") in local_ids:
            continue
        block = _context_block(
            f"Earlier same-post comment {comment.get('_comment_id')}",
            comment.get("_body", ""),
            max_chars=600,
        )
        if block:
            broader_blocks.append(block)
    if broader_blocks:
        sections.append(
            "Broader earlier-thread context. These comments appeared earlier in the same post and are included only when they can help identify a repeated issue, entity, question, or reference. "
            + " ".join(broader_blocks)
        )

    expanded = "\n\n".join(section for section in sections if section).strip()

    if word_count(expanded) < min_words:
        limited_source_note = (
            "Additional extraction note. The available supplied context is shorter than the requested minimum, so the system expands the report by documenting the source boundaries rather than inventing new facts. "
            "Only prior post text, parent text, ancestor text, sibling text, earlier-thread text, and deterministic weighted context candidates are used. "
            "The target comment itself is not treated as prior context. "
            "This expanded text is a context-extraction record, not a judgment about the writer, the factual truth of the claim, or the social meaning of the reply. "
            "When the post is available, it is considered a high-priority source because it anchors the shared topic for the thread. "
            "When a direct parent is available, it is considered the closest local source. "
            "When ancestor, sibling, or broader-thread comments are available, they are included only as possible referential sources. "
        )
        while word_count(expanded) < min_words:
            expanded = (expanded + "\n\n" + limited_source_note).strip()

    return expanded

def summarize_text(text: str, max_chars: int = 220) -> str:
    """Return a first-sentence summary or compact excerpt."""
    value = normalize_text(text)
    if not value:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", value)
    if sentences and len(sentences[0]) <= max_chars:
        return sentences[0]

    return excerpt(value, max_chars=max_chars)


def quoted_texts(text: str) -> List[str]:
    """Return markdown quote lines from a comment."""
    return [
        normalize_text(match.group(1))
        for match in QUOTE_LINE_RE.finditer(normalize_text(text))
        if normalize_text(match.group(1))
    ]


def starts_with_discourse_marker(text: str) -> bool:
    """Detect markers that often depend on immediately prior context."""
    return bool(
        re.match(
            r"^\s*(yes|yeah|yep|no|nope|nah|same|exactly|also|but|however|why|source|agreed|true|right|this|that)\b",
            normalize_text(text).lower(),
        )
    )


def has_vague_reference(text: str) -> bool:
    """Return True if text contains context-dependent reference expressions."""
    return any(pattern.search(text or "") for _, pattern in REFERENCE_PATTERNS)


def detect_reference_expressions(text: str) -> List[str]:
    """Detect vague or context-dependent expressions in target text."""
    found = []
    for expression, pattern in REFERENCE_PATTERNS:
        if pattern.search(text or ""):
            found.append(expression)
    return found


def extract_post_context(post: Optional[Mapping[str, Any]]) -> str:
    """Extract global post context from title and body."""
    if not post:
        return "Post context unavailable."

    title = normalize_text(post.get("_title", ""))
    body = normalize_text(post.get("_body", ""))

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if body:
        parts.append(f"Body: {excerpt(body, 500)}")
    else:
        parts.append("Post body is missing or empty.")

    return " ".join(parts).strip() or "Post context unavailable."


def extract_local_context(
    target_comment: Mapping[str, Any],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
) -> str:
    """Extract nearby conversational context around a target comment."""
    pieces = []

    if ancestors:
        chain = " -> ".join(
            f"{ancestor.get('_comment_id')}: {excerpt(ancestor.get('_body', ''), 120)}"
            for ancestor in ancestors
        )
        pieces.append(f"Ancestor chain: {chain}")

    if parent:
        pieces.append(f"Direct parent: {excerpt(parent.get('_body', ''), 220)}")
    else:
        pieces.append("Direct parent: none; target is a top-level reply or parent is unavailable.")

    useful_siblings = [sibling for sibling in prior_siblings if not is_deleted_or_empty(sibling.get("_body", ""))]
    if useful_siblings:
        sibling_text = " | ".join(
            f"{sibling.get('_comment_id')}: {excerpt(sibling.get('_body', ''), 140)}"
            for sibling in useful_siblings[-3:]
        )
        pieces.append(f"Earlier siblings: {sibling_text}")

    return " ".join(pieces).strip()


def _source_text(source: Mapping[str, Any]) -> str:
    """Return the text field appropriate to a context source."""
    if source.get("_source_type") == "post_title":
        return normalize_text(source.get("_title", ""))
    if source.get("_source_type") == "post_body":
        return normalize_text(source.get("_body", ""))
    return normalize_text(source.get("_body", ""))


def _source_id(source: Mapping[str, Any]) -> Any:
    """Return source ID for output."""
    if source.get("_source_type") in {"post_title", "post_body"}:
        return source.get("_post_id")
    return source.get("_comment_id", source.get("_post_id"))


def _candidate_sources(
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Record]:
    """Build ordered context candidates for reference resolution."""
    candidates: List[Record] = []

    if parent:
        item = dict(parent)
        item["_source_type"] = "direct_parent"
        candidates.append(item)

    for ancestor in reversed(list(ancestors)):
        item = dict(ancestor)
        item["_source_type"] = "ancestor_comment"
        candidates.append(item)

    for sibling in reversed(list(prior_siblings[-5:])):
        item = dict(sibling)
        item["_source_type"] = "prior_sibling"
        candidates.append(item)

    if post:
        if normalize_text(post.get("_title", "")):
            item = dict(post)
            item["_source_type"] = "post_title"
            candidates.append(item)
        if normalize_text(post.get("_body", "")):
            item = dict(post)
            item["_source_type"] = "post_body"
            candidates.append(item)

    if prior_thread_comments:
        for comment in reversed(list(prior_thread_comments[-8:])):
            item = dict(comment)
            item["_source_type"] = "prior_thread_comment"
            candidates.append(item)

    return [candidate for candidate in candidates if normalize_text(_source_text(candidate))]


def _best_context_candidate(
    target_text: str,
    candidates: Sequence[Mapping[str, Any]],
    preferred_source: Optional[str] = None,
) -> Tuple[Optional[Mapping[str, Any]], float]:
    """Select a likely antecedent/source for a vague expression."""
    if not candidates:
        return None, 0.0

    source_base = {
        "direct_parent": 0.80,
        "ancestor_comment": 0.62,
        "prior_sibling": 0.55,
        "post_title": 0.50,
        "post_body": 0.45,
        "prior_thread_comment": 0.30,
    }

    target_tokens = token_set(target_text)
    best: Optional[Mapping[str, Any]] = None
    best_score = -1.0

    for index, candidate in enumerate(candidates):
        source_type = str(candidate.get("_source_type", "unknown"))
        base = source_base.get(source_type, 0.25)
        proximity = max(0.0, 1.0 - index * 0.06)
        overlap = lexical_overlap(target_text, _source_text(candidate))

        score = base * 0.55 + proximity * 0.25 + overlap * 0.20

        if preferred_source and source_type == preferred_source:
            score += 0.18

        if len(target_tokens) <= 2 and source_type == "direct_parent":
            score += 0.15

        if score > best_score:
            best = candidate
            best_score = score

    return best, max(0.0, min(1.0, best_score))


def resolve_references(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolve vague references in the target comment using prior context."""
    target_text = normalize_text(target_comment.get("_body", ""))
    expressions = detect_reference_expressions(target_text)

    resolved: List[Dict[str, Any]] = []
    ambiguities: List[str] = []

    if not expressions:
        return resolved, ambiguities

    candidates = _candidate_sources(post, parent, ancestors, prior_siblings, prior_thread_comments)

    for expression in expressions:
        preferred_source = None

        if expression in {"OP", "the title"}:
            preferred_source = "post_title"
        elif expression in {"the article", "the study", "the video"}:
            preferred_source = "post_body"
        elif expression == "the parent comment":
            preferred_source = "direct_parent"
        elif expression in {"yes", "no", "agreed", "exactly", "same", "why", "source", "what do you mean"}:
            preferred_source = "direct_parent"

        best, score = _best_context_candidate(target_text, candidates, preferred_source=preferred_source)

        if best is None:
            resolved.append(
                {
                    "expression": expression,
                    "resolved_as": "",
                    "source": "unknown",
                    "source_id": None,
                    "confidence": 0.0,
                }
            )
            ambiguities.append(
                f"Reference '{expression}' could not be resolved because no suitable prior context was available."
            )
            continue

        confidence = round(max(0.0, min(1.0, score)), 2)
        if confidence < 0.45:
            ambiguities.append(
                f"Reference '{expression}' has multiple or weak possible antecedents; selected source is low confidence."
            )

        resolved.append(
            {
                "expression": expression,
                "resolved_as": excerpt(_source_text(best), 220),
                "source": str(best.get("_source_type", "unknown")),
                "source_id": _source_id(best),
                "confidence": confidence,
            }
        )

    return resolved, ambiguities


def _make_weight_item(
    source: str,
    item_id: Any,
    weight: float,
    text: str,
    evidence: str,
) -> Dict[str, Any]:
    """Create a weighted context output item."""
    return {
        "source": source,
        "id": item_id,
        "relevance_weight": round(max(0.0, min(1.0, weight)), 2),
        "text_excerpt": excerpt(text, 260),
        "evidence_summary": evidence,
    }


def compute_context_weights(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Assign relevance weights to context items.

    Explicit weighting logic:
    - target_comment: always 1.00
    - direct_parent: usually 0.80-1.00 when available
    - post_title/post_body: higher for top-level replies, moderate for nested replies
    - ancestor_comment: decays with distance from target
    - prior_sibling: moderate when nearby, overlapping, or needed for vague references
    - prior_thread_comment: low weight for recurring themes outside the local chain
    - later comments: not passed in and never used
    """
    target_text = normalize_text(target_comment.get("_body", ""))
    weighted: List[Dict[str, Any]] = [
        _make_weight_item(
            "target_comment",
            target_comment.get("_comment_id"),
            1.00,
            target_text,
            "The target comment whose response context is being inferred.",
        )
    ]

    target_has_vague_reference = has_vague_reference(target_text) or starts_with_discourse_marker(target_text)

    if parent:
        parent_text = normalize_text(parent.get("_body", ""))
        overlap = lexical_overlap(target_text, parent_text)
        parent_weight = 0.82 + min(0.12, overlap * 0.25)

        if target_has_vague_reference or len(token_set(target_text)) <= 4:
            parent_weight += 0.08
        if is_deleted_or_empty(parent_text):
            parent_weight -= 0.25

        weighted.append(
            _make_weight_item(
                "direct_parent",
                parent.get("_comment_id"),
                parent_weight,
                parent_text,
                "Direct reply target; weighted highly because the target comment is nested under it.",
            )
        )

    if post:
        title = normalize_text(post.get("_title", ""))
        body = normalize_text(post.get("_body", ""))
        is_top_level = parent is None

        if title:
            title_weight = POST_CONTEXT_TOP_LEVEL_TITLE_BASE if is_top_level else POST_CONTEXT_NESTED_TITLE_BASE
            title_weight += POST_CONTEXT_PRIORITY_BOOST
            title_weight += min(0.22, lexical_overlap(target_text, title) * 0.40)
            if re.search(r"\b(title|op|article|study|video|thread|submission)\b|\b(the|this|original)\s+post\b", target_text, re.I):
                title_weight += 0.18

            weighted.append(
                _make_weight_item(
                    "post_title",
                    post.get("_post_id"),
                    title_weight,
                    title,
                    "Global post title; weighted more when target is top-level or references the title.",
                )
            )

        if body:
            body_weight = POST_CONTEXT_TOP_LEVEL_BODY_BASE if is_top_level else POST_CONTEXT_NESTED_BODY_BASE
            body_weight += POST_CONTEXT_PRIORITY_BOOST
            body_weight += min(0.22, lexical_overlap(target_text, body) * 0.40)
            if re.search(r"\b(article|study|video|op|thread|submission|title)\b|\b(the|this|original)\s+post\b", target_text, re.I):
                body_weight += 0.18

            weighted.append(
                _make_weight_item(
                    "post_body",
                    post.get("_post_id"),
                    body_weight,
                    body,
                    "Global post body; weighted more when target is top-level or references post-level material.",
                )
            )

    for distance, ancestor in enumerate(reversed(list(ancestors)), start=1):
        ancestor_text = normalize_text(ancestor.get("_body", ""))
        ancestor_weight = 0.78 - 0.08 * (distance - 1)
        ancestor_weight += min(0.12, lexical_overlap(target_text, ancestor_text) * 0.25)

        if target_has_vague_reference:
            ancestor_weight += 0.04
        if is_deleted_or_empty(ancestor_text):
            ancestor_weight -= 0.25

        weighted.append(
            _make_weight_item(
                "ancestor_comment",
                ancestor.get("_comment_id"),
                ancestor_weight,
                ancestor_text,
                f"Ancestor in the reply chain, distance {distance} from the direct-parent side.",
            )
        )

    for rank_from_target, sibling in enumerate(reversed(prior_siblings[-6:]), start=1):
        sibling_text = normalize_text(sibling.get("_body", ""))
        proximity = max(0.0, 1.0 - 0.12 * (rank_from_target - 1))
        sibling_weight = 0.28 + 0.20 * proximity + 0.25 * lexical_overlap(target_text, sibling_text)

        if target_has_vague_reference:
            sibling_weight += 0.08
        if is_deleted_or_empty(sibling_text):
            sibling_weight -= 0.20

        if sibling_weight >= 0.25:
            weighted.append(
                _make_weight_item(
                    "prior_sibling",
                    sibling.get("_comment_id"),
                    sibling_weight,
                    sibling_text,
                    "Earlier sibling under the same parent; may resolve elliptical replies or references.",
                )
            )

    local_ids = {target_comment.get("_comment_id")}
    if parent:
        local_ids.add(parent.get("_comment_id"))
    local_ids.update(ancestor.get("_comment_id") for ancestor in ancestors)
    local_ids.update(sibling.get("_comment_id") for sibling in prior_siblings)

    thread_candidates: List[Tuple[float, Mapping[str, Any]]] = []
    for index, prior_comment in enumerate(prior_thread_comments):
        if prior_comment.get("_comment_id") in local_ids:
            continue

        prior_text = normalize_text(prior_comment.get("_body", ""))
        if is_deleted_or_empty(prior_text):
            continue

        overlap = lexical_overlap(target_text, prior_text)
        if overlap <= 0 and not target_has_vague_reference:
            continue

        recency_rank = len(prior_thread_comments) - index
        recency_bonus = min(0.10, 1.0 / max(1, recency_rank + 4))
        weight = 0.10 + min(0.24, overlap * 0.45) + recency_bonus

        if weight >= 0.13:
            thread_candidates.append((weight, prior_comment))

    thread_candidates.sort(key=lambda item: item[0], reverse=True)
    for weight, prior_comment in thread_candidates[:5]:
        weighted.append(
            _make_weight_item(
                "prior_thread_comment",
                prior_comment.get("_comment_id"),
                weight,
                normalize_text(prior_comment.get("_body", "")),
                "Earlier comment elsewhere in the same post; low-weight evidence for recurring thread context.",
            )
        )

    weighted.sort(key=lambda item: item["relevance_weight"], reverse=True)
    return weighted


def _top_weight_by_source(weighted_context: Sequence[Mapping[str, Any]], source: str) -> float:
    """Return the highest relevance weight for a source type."""
    weights = [float(item.get("relevance_weight", 0.0)) for item in weighted_context if item.get("source") == source]
    return max(weights) if weights else 0.0


def _is_probable_topic_shift(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
) -> bool:
    """Detect whether the target introduces a new but related issue."""
    target_text = normalize_text(target_comment.get("_body", ""))
    if not target_text or len(token_set(target_text)) < 4:
        return False

    parent_text = normalize_text(parent.get("_body", "")) if parent else ""
    post_text = ""
    if post:
        post_text = f"{post.get('_title', '')} {post.get('_body', '')}"

    has_shift_marker = bool(
        re.search(
            r"\b(also|another|separate|different|related|speaking of|while we're|on another note|what about)\b",
            target_text,
            re.I,
        )
    )

    return has_shift_marker and lexical_overlap(target_text, parent_text) < 0.12 and lexical_overlap(target_text, post_text) >= 0.05


def _recurring_thread_score(
    target_comment: Mapping[str, Any],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    exclude_ids: Optional[set] = None,
) -> float:
    """Estimate whether target refers to a repeated theme across earlier comments."""
    exclude_ids = exclude_ids or set()
    target_text = normalize_text(target_comment.get("_body", ""))

    if not token_set(target_text):
        return 0.0

    hits = 0
    total_overlap = 0.0

    for comment in prior_thread_comments:
        if comment.get("_comment_id") in exclude_ids:
            continue

        prior_text = normalize_text(comment.get("_body", ""))
        if is_deleted_or_empty(prior_text):
            continue

        overlap = lexical_overlap(target_text, prior_text)
        if overlap >= 0.12:
            hits += 1
            total_overlap += overlap

    if hits < 2:
        return 0.0

    return min(1.0, 0.20 * hits + total_overlap / max(1, hits))


def classify_context_type(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    weighted_context: Optional[Sequence[Mapping[str, Any]]] = None,
    resolved_references: Optional[Sequence[Mapping[str, Any]]] = None,
    ambiguities: Optional[Sequence[str]] = None,
) -> str:
    """Classify context type only; does not classify stance."""
    target_text = normalize_text(target_comment.get("_body", ""))
    if is_deleted_or_empty(target_text):
        return "ambiguous"

    if weighted_context is None:
        weighted_context = compute_context_weights(
            target_comment,
            post,
            parent,
            ancestors,
            prior_siblings,
            prior_thread_comments,
        )

    resolved_references = resolved_references or []

    if len(token_set(target_text)) <= 1 and has_vague_reference(target_text):
        return "ambiguous"

    sibling_ref_score = max(
        [float(ref.get("confidence", 0.0)) for ref in resolved_references if ref.get("source") == "prior_sibling"],
        default=0.0,
    )
    if sibling_ref_score >= 0.50:
        return "prior_sibling_reference"

    local_ids = {target_comment.get("_comment_id")}
    if parent:
        local_ids.add(parent.get("_comment_id"))
    local_ids.update(ancestor.get("_comment_id") for ancestor in ancestors)
    local_ids.update(sibling.get("_comment_id") for sibling in prior_siblings)

    if _recurring_thread_score(target_comment, prior_thread_comments, exclude_ids=local_ids) >= 0.45:
        return "broader_thread_reference"

    if _is_probable_topic_shift(target_comment, post, parent):
        return "topic_shift"

    if parent is None:
        return "post_direct"

    parent_weight = _top_weight_by_source(weighted_context, "direct_parent")
    title_weight = _top_weight_by_source(weighted_context, "post_title")
    body_weight = _top_weight_by_source(weighted_context, "post_body")
    ancestor_weight = _top_weight_by_source(weighted_context, "ancestor_comment")
    sibling_weight = _top_weight_by_source(weighted_context, "prior_sibling")

    ancestor_ref = any(
        ref.get("source") == "ancestor_comment" and float(ref.get("confidence", 0.0)) >= 0.50
        for ref in resolved_references
    )

    if ancestors and (ancestor_ref or (ancestor_weight >= 0.66 and parent_weight < 0.86)):
        return "ancestor_chain"

    if sibling_weight >= 0.58 and starts_with_discourse_marker(target_text):
        return "prior_sibling_reference"

    post_weight = max(title_weight, body_weight)
    post_reference_signal = bool(
        re.search(
            r"\b(op|article|study|video|title|thread|submission)\b|\b(the|this|original)\s+post\b",
            target_text,
            re.I,
        )
    )

    # Post-level context is intentionally prioritized more strongly than before,
    # but vague pronouns alone should not override a strong direct-parent signal.
    if post_reference_signal and post_weight >= 0.68:
        return "post_direct"

    if post_weight >= 0.78 and parent_weight < 0.82:
        return "post_direct"

    if parent_weight >= 0.70 and parent_weight >= post_weight + 0.06:
        return "parent_reply"

    if post_weight >= 0.64 and parent_weight < 0.74:
        return "post_direct"

    if parent_weight >= 0.60:
        return "parent_reply"

    if post_weight >= 0.58:
        return "post_direct"

    return "ambiguous"


def _extract_issue_from_source(text: str) -> str:
    """Extract a compact claim, issue, or question from a source text."""
    normalized = normalize_text(text)
    if not normalized:
        return ""

    quotes = quoted_texts(normalized)
    if quotes:
        return excerpt(quotes[0], 220)

    question_sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", normalized) if "?" in sentence]
    if question_sentences:
        return excerpt(question_sentences[0], 220)

    return summarize_text(normalized, max_chars=220)


def fallback_extract_target_context_with_heuristics(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    context_type: Optional[str] = None,
    resolved_references: Optional[Sequence[Mapping[str, Any]]] = None,
    weighted_context: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Infer target-specific context with transparent heuristics."""
    target_text = normalize_text(target_comment.get("_body", ""))

    if is_deleted_or_empty(target_text):
        return "The target comment is deleted or empty, so its specific response context cannot be inferred."

    if resolved_references is None:
        resolved_references, _ = resolve_references(
            target_comment,
            post,
            parent,
            ancestors,
            prior_siblings,
            prior_thread_comments,
        )

    if weighted_context is None:
        weighted_context = compute_context_weights(
            target_comment,
            post,
            parent,
            ancestors,
            prior_siblings,
            prior_thread_comments,
        )

    if context_type is None:
        context_type = classify_context_type(
            target_comment,
            post,
            parent,
            ancestors,
            prior_siblings,
            prior_thread_comments,
            weighted_context=weighted_context,
            resolved_references=resolved_references,
        )

    resolved_high_confidence = [
        ref for ref in resolved_references if float(ref.get("confidence", 0.0)) >= 0.50
    ]
    if resolved_high_confidence:
        best_ref = max(resolved_high_confidence, key=lambda ref: float(ref.get("confidence", 0.0)))
        return (
            "The target comment appears to respond to the referenced context "
            f"'{best_ref.get('expression')}', most likely: {best_ref.get('resolved_as')}."
        )

    if context_type == "post_direct":
        title = normalize_text(post.get("_title", "")) if post else ""
        body = normalize_text(post.get("_body", "")) if post else ""

        if title and body:
            return f"The target comment appears to respond directly to the post topic: {excerpt(title, 160)}; {excerpt(body, 220)}"
        if title:
            return f"The target comment appears to respond directly to the post title: {excerpt(title, 220)}"
        if body:
            return f"The target comment appears to respond directly to the post body: {excerpt(body, 260)}"
        return "The target comment appears to respond to the post, but the post text is unavailable."

    if context_type == "parent_reply":
        if parent:
            issue = _extract_issue_from_source(parent.get("_body", ""))
            return f"The target comment appears to respond to the direct parent comment's point: {issue}"
        return "The target comment appears to be a reply, but the direct parent comment is unavailable."

    if context_type == "ancestor_chain":
        chain_bits = []
        for ancestor in ancestors:
            issue = _extract_issue_from_source(ancestor.get("_body", ""))
            if issue:
                chain_bits.append(issue)
        if parent:
            parent_issue = _extract_issue_from_source(parent.get("_body", ""))
            if parent_issue:
                chain_bits.append(parent_issue)

        if chain_bits:
            return "The target comment appears to depend on the reply-chain context: " + " / ".join(chain_bits[-3:])
        return "The target comment appears to depend on multiple prior comments in its ancestor chain."

    if context_type == "prior_sibling_reference":
        candidates = [item for item in weighted_context if item.get("source") == "prior_sibling"]
        if candidates:
            best = max(candidates, key=lambda item: float(item.get("relevance_weight", 0.0)))
            return f"The target comment appears to refer to an earlier sibling comment: {best.get('text_excerpt')}"
        return "The target comment appears to refer to an earlier sibling comment under the same parent."

    if context_type == "broader_thread_reference":
        candidates = [item for item in weighted_context if item.get("source") == "prior_thread_comment"]
        if candidates:
            summaries = [candidate.get("text_excerpt", "") for candidate in candidates[:3]]
            return "The target comment appears to refer to a recurring earlier thread theme: " + " | ".join(summaries)
        return "The target comment appears to refer to a broader recurring discussion in earlier comments."

    if context_type == "topic_shift":
        post_topic = ""
        if post:
            post_topic = normalize_text(post.get("_title", "")) or excerpt(post.get("_body", ""), 160)
        if post_topic:
            return f"The target comment appears to introduce a related new issue within the post topic: {excerpt(post_topic, 220)}"
        return "The target comment appears to introduce a related new issue, but the post topic is weakly available."

    if parent:
        return f"The target context is ambiguous; the nearest available context is the direct parent: {excerpt(parent.get('_body', ''), 240)}"

    return "The target context is ambiguous and cannot be confidently inferred from available prior context."


def llm_generate(prompt: str) -> str:
    """
    Optional local LLM hook.

    Replace this function with a local model call if desired. By default it
    returns an empty string, causing the heuristic extractor to be used.
    """
    return ""


def _json_default(value: Any) -> str:
    """JSON serializer fallback used only for prompt construction."""
    return str(value)


def _prompt_comment_item(
    comment: Optional[Mapping[str, Any]],
    source: str,
    max_chars: int = 360,
) -> Optional[Dict[str, Any]]:
    """Convert an internal comment record into a compact prompt item."""
    if not comment:
        return None

    return {
        "source": source,
        "comment_id": comment.get("_comment_id"),
        "parent_id": comment.get("_parent_id"),
        "post_id": comment.get("_post_id"),
        "author": comment.get("_author"),
        "depth": comment.get("_depth"),
        "text": excerpt(comment.get("_body", ""), max_chars),
    }


def _prompt_post_item(post: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert an internal post record into a compact prompt item."""
    if not post:
        return None

    return {
        "source": "post",
        "post_id": post.get("_post_id"),
        "author": post.get("_author"),
        "title": excerpt(post.get("_title", ""), 260),
        "body": excerpt(post.get("_body", ""), 700),
    }


def _context_candidate_texts(
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    weighted_context: Sequence[Mapping[str, Any]],
) -> List[str]:
    """
    Collect supplied prior text candidates for validating extracted context.

    These are the only texts from which the LLM judge is allowed to extract
    target_context. The target comment itself is intentionally excluded.
    """
    candidates: List[str] = []

    if post:
        candidates.extend([
            normalize_text(post.get("_title", "")),
            normalize_text(post.get("_body", "")),
        ])

    for item in [parent, *ancestors, *prior_siblings, *prior_thread_comments]:
        if item:
            candidates.append(normalize_text(item.get("_body", "")))

    for item in weighted_context:
        source = normalize_text(item.get("source", ""))
        if source != "target_comment":
            candidates.append(normalize_text(item.get("text_excerpt", "")))

    return [
        candidate
        for candidate in dedupe_preserve_order(candidates)
        if candidate
    ]


def _content_tokens(text: Any) -> List[str]:
    """Return simple content tokens for extractive-overlap validation."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{1,}", normalize_text(text).lower())
    return [
        token.strip("'-")
        for token in tokens
        if token.strip("'-") and token.strip("'-") not in STOPWORDS
    ]


def _has_extractive_overlap(
    extracted_text: str,
    candidate_texts: Sequence[str],
    min_overlap: float = 0.22,
) -> bool:
    """
    Check whether target_context is grounded in supplied prior text.

    The check supports longer extractive reports by validating against both
    individual candidates and the aggregate set of all supplied prior texts.
    """
    extracted = normalize_text(extracted_text)
    if not extracted:
        return False

    extracted_lower = extracted.lower()
    extracted_tokens = set(_content_tokens(extracted))
    if not extracted_tokens:
        return False

    aggregate_tokens = set()
    for candidate in candidate_texts:
        candidate_norm = normalize_text(candidate)
        if not candidate_norm:
            continue

        if extracted_lower in candidate_norm.lower():
            return True

        candidate_tokens = set(_content_tokens(candidate_norm))
        aggregate_tokens.update(candidate_tokens)

        if not candidate_tokens:
            continue

        overlap = len(extracted_tokens.intersection(candidate_tokens)) / max(1, len(extracted_tokens))
        if overlap >= min_overlap:
            return True

        candidate_coverage = len(extracted_tokens.intersection(candidate_tokens)) / max(1, len(candidate_tokens))
        if candidate_coverage >= 0.50 and len(extracted_tokens.intersection(candidate_tokens)) >= 4:
            return True

    if aggregate_tokens:
        aggregate_overlap = len(extracted_tokens.intersection(aggregate_tokens)) / max(1, len(extracted_tokens))
        if aggregate_overlap >= min_overlap:
            return True

    return False



def build_llm_judge_prompt(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    weighted_context: Sequence[Mapping[str, Any]],
    resolved_references: Sequence[Mapping[str, Any]],
    heuristic_context_type: str,
    heuristic_target_context: str,
    max_prior_siblings: int = 6,
    max_prior_thread_comments: int = 10,
) -> str:
    """
    Build a structured LLM-judge prompt for extractive context resolution.

    The LLM is asked only to extract the prior text span(s) that the target
    comment is responding to. It must not classify stance, sentiment, toxicity,
    ideology, correctness, agreement, disagreement, support, or opposition.
    """
    payload = {
        "task": "extractive_conversational_context_resolution_only",
        "allowed_context_types": sorted(CONTEXT_TYPES),
        "target_comment": _prompt_comment_item(target_comment, "target_comment", max_chars=900),
        "post_context": _prompt_post_item(post),
        "direct_parent": _prompt_comment_item(parent, "direct_parent", max_chars=700),
        "ancestor_chain_top_to_bottom": [
            _prompt_comment_item(ancestor, "ancestor_comment", max_chars=520)
            for ancestor in ancestors
        ],
        "prior_siblings_earliest_to_latest": [
            _prompt_comment_item(sibling, "prior_sibling", max_chars=420)
            for sibling in list(prior_siblings)[-max_prior_siblings:]
        ],
        "earlier_thread_comments_earliest_to_latest": [
            _prompt_comment_item(comment, "prior_thread_comment", max_chars=360)
            for comment in list(prior_thread_comments)[-max_prior_thread_comments:]
        ],
        "heuristic_context_type": heuristic_context_type,
        "heuristic_target_context": heuristic_target_context,
        "heuristic_resolved_references": list(resolved_references),
        "heuristic_weighted_context": list(weighted_context)[:12],
        "minimum_target_context_words": MIN_TARGET_CONTEXT_WORDS,
        "post_context_priority": (
            "Post title and post body should receive high priority when they plausibly anchor "
            "the target comment's issue, entity, question, joke, or reference."
        ),
    }

    schema = {
        "context_type": "one of allowed_context_types",
        "target_context": (
            "an extended extractive context report of at least 300 words when enough supplied prior text is available; "
            "copy or closely preserve prior text spans from supplied context; "
            "if multiple spans are needed, join and explain source boundaries; do not output classification labels"
        ),
        "resolved_references": {
            "this": "resolved prior text or null",
            "that": "resolved prior text or null",
            "it": "resolved prior text or null",
            "they": "resolved prior text or null",
            "op": "resolved prior text or null",
            "article": "resolved prior text or null",
        },
        "confidence": "number between 0 and 1 for context text extraction only",
        "ambiguities": ["uncertainty notes about text-span resolution only"],
    }

    return (
        "You are an LLM judge for Reddit-style TEXT EXTRACTION.\n"
        "Your only job is to extract the prior conversational text span(s) that the TARGET COMMENT is responding to.\n\n"
        "Extractive focus:\n"
        "- target_context must be an extended extractive context report, not a stance or sentiment analysis.\n"
        "- The final target_context should be at least 300 words when enough supplied prior text exists.\n"
        "- Prefer verbatim wording from the post title, post body, direct parent, ancestor, prior sibling, or earlier thread comment.\n"
        "- Give post title and post body higher priority when they plausibly anchor the issue, entity, question, joke, or reference being answered.\n"
        "- Do not summarize the target author's attitude. Do not classify the relationship between comments.\n"
        "- If the exact span is unclear, return the closest supplied text span and add an ambiguity note.\n\n"
        "Strict prohibitions:\n"
        "- Do NOT classify stance, sentiment, toxicity, morality, ideology, correctness, agreement, disagreement, support, opposition, or political position.\n"
        "- Do NOT decide whether the target is right or wrong.\n"
        "- Do NOT infer facts not present in the supplied context.\n"
        "- Do NOT use labels such as stance, agree, disagree, support, oppose, sentiment, toxicity, pro, anti, or political position.\n\n"
        "Use only the supplied post, parent, ancestor, sibling, earlier-thread, and heuristic context items.\n"
        "Preserve uncertainty. If multiple prior texts are plausible, choose 'ambiguous' or include ambiguity notes.\n"
        "A top-level comment usually extracts context from the post. A nested comment may extract context from its direct parent, but the post context should be preferred when the target wording plausibly refers back to the post-level issue or entity.\n\n"
        "Return valid JSON only. Do not include markdown, comments, explanations, or text outside the JSON object.\n"
        "Required JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Structured input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)}"
    )


def build_llm_prompt(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
) -> str:
    """
    Backward-compatible wrapper around the JSON LLM-judge prompt builder.

    New code should call build_llm_judge_prompt directly so it can pass
    heuristic context, resolved references, and weighted context.
    """
    resolved_references, _ = resolve_references(
        target_comment,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
    )
    weighted_context = compute_context_weights(
        target_comment,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
    )
    heuristic_context_type = classify_context_type(
        target_comment,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
        weighted_context=weighted_context,
        resolved_references=resolved_references,
        ambiguities=[],
    )
    heuristic_target_context = fallback_extract_target_context_with_heuristics(
        target_comment,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
        context_type=heuristic_context_type,
        resolved_references=resolved_references,
        weighted_context=weighted_context,
    )

    return build_llm_judge_prompt(
        target_comment=target_comment,
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        weighted_context=weighted_context,
        resolved_references=resolved_references,
        heuristic_context_type=heuristic_context_type,
        heuristic_target_context=heuristic_target_context,
    )


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from a string.

    This makes local model outputs more robust when they accidentally include
    leading/trailing prose or markdown fences.
    """
    value = normalize_text(text)
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)

    start = value.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(value)):
        char = value[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]

    return ""


def _contains_banned_llm_label(value: Any) -> bool:
    """
    Return True if an LLM output contains prohibited classification labels.

    The check is intentionally strict. Invalid outputs are discarded and the
    deterministic heuristic fallback is used instead.
    """
    if isinstance(value, Mapping):
        return any(
            LLM_JUDGE_BANNED_LABEL_RE.search(str(key)) or _contains_banned_llm_label(item)
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple)):
        return any(_contains_banned_llm_label(item) for item in value)

    if isinstance(value, str):
        return bool(LLM_JUDGE_BANNED_LABEL_RE.search(value))

    return False


def _as_probability(value: Any) -> Optional[float]:
    """Convert a value to a valid probability in [0, 1], or return None."""
    try:
        probability = float(value)
    except Exception:
        return None

    if 0.0 <= probability <= 1.0:
        return probability
    return None


def _validate_resolved_references(value: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Validate and sanitize LLM-produced resolved-reference objects.

    The preferred prompt schema uses a compact dict such as
    {"this": "...", "that": null, "op": "..."}.
    For backward compatibility, a list of richer reference objects is also
    accepted and normalized into the same inspectable list format used by the
    heuristic pipeline.
    """
    sanitized: List[Dict[str, Any]] = []

    if isinstance(value, Mapping):
        allowed_keys = {"this", "that", "it", "they", "them", "op", "article", "study", "video", "title", "same", "exactly"}
        for expression, resolved_as in value.items():
            expression_norm = normalize_text(expression).lower()
            if expression_norm not in allowed_keys:
                return None

            resolved_text = normalize_text(resolved_as)
            if not resolved_text:
                continue

            sanitized.append(
                {
                    "expression": expression_norm,
                    "resolved_as": excerpt(resolved_text, 260),
                    "source": "unknown",
                    "source_id": None,
                    "confidence": 0.50,
                }
            )
        return sanitized

    if not isinstance(value, list):
        return None

    required = {"expression", "resolved_as", "source", "source_id", "confidence"}

    for item in value:
        if not isinstance(item, Mapping):
            return None
        if not required.issubset(item.keys()):
            return None

        confidence = _as_probability(item.get("confidence"))
        if confidence is None:
            return None

        source = normalize_text(item.get("source", "")).lower()
        if source not in LLM_REFERENCE_SOURCES:
            return None

        sanitized.append(
            {
                "expression": normalize_text(item.get("expression", "")),
                "resolved_as": excerpt(item.get("resolved_as", ""), 260),
                "source": source,
                "source_id": item.get("source_id"),
                "confidence": round(confidence, 2),
            }
        )

    return sanitized



def validate_llm_judge_output(
    payload: Mapping[str, Any],
    candidate_texts: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate and sanitize an LLM judge output.

    If candidate_texts are supplied, target_context must be extractive enough
    to overlap with prior text supplied to the LLM. This keeps the LLM focused
    on text extraction rather than free-form interpretation.
    """
    if not isinstance(payload, Mapping):
        return None

    if not LLM_JUDGE_REQUIRED_FIELDS.issubset(payload.keys()):
        return None

    if _contains_banned_llm_label(payload):
        return None

    context_type = normalize_text(payload.get("context_type", "")).lower()
    if context_type not in CONTEXT_TYPES:
        return None

    target_context = normalize_text(payload.get("target_context", ""))
    if not target_context or LOW_QUALITY_CONTEXT_RE.match(target_context):
        return None

    if candidate_texts and not _has_extractive_overlap(target_context, candidate_texts):
        return None

    confidence = _as_probability(payload.get("confidence"))
    if confidence is None:
        return None

    resolved_references = _validate_resolved_references(payload.get("resolved_references"))
    if resolved_references is None:
        return None

    ambiguities_raw = payload.get("ambiguities")
    if not isinstance(ambiguities_raw, list):
        return None

    ambiguities = [
        normalize_text(item)
        for item in ambiguities_raw
        if normalize_text(item)
    ]

    return {
        "context_type": context_type,
        "target_context": target_context,
        "resolved_references": resolved_references,
        "confidence": round(confidence, 2),
        "ambiguities": dedupe_preserve_order(ambiguities),
    }


def parse_llm_judge_response(
    response: str,
    candidate_texts: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse and validate an LLM judge response.

    The response must contain a valid JSON object matching the required schema.
    Outputs containing prohibited stance/sentiment/toxicity/support/opposition
    labels are rejected. When candidate_texts are provided, target_context must
    overlap with text that was supplied to the model.
    """
    if not normalize_text(response):
        return None

    json_text = _extract_first_json_object(response)
    if not json_text:
        return None

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    return validate_llm_judge_output(payload, candidate_texts=candidate_texts)


def extract_target_context_with_llm_judge(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    weighted_context: Sequence[Mapping[str, Any]],
    resolved_references: Sequence[Mapping[str, Any]],
    heuristic_context_type: str,
    heuristic_target_context: str,
    llm_generate_fn: Callable[[str], str],
) -> Optional[Dict[str, Any]]:
    """
    Run the injectable LLM judge and return validated context extraction output.

    The LLM is used only as an interpreter of the deterministic context bundle.
    If the model call fails, returns invalid JSON, includes banned labels, fails
    schema validation, or does not extract text from supplied context, None is
    returned and the caller should use the heuristic fallback.
    """
    prompt = build_llm_judge_prompt(
        target_comment=target_comment,
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        weighted_context=weighted_context,
        resolved_references=resolved_references,
        heuristic_context_type=heuristic_context_type,
        heuristic_target_context=heuristic_target_context,
    )

    try:
        response = llm_generate_fn(prompt)
    except Exception:
        return None

    candidate_texts = _context_candidate_texts(
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        weighted_context=weighted_context,
    )

    parsed = parse_llm_judge_response(response, candidate_texts=candidate_texts)
    if parsed is None:
        return None

    parsed["extraction_method"] = "llm_judge"
    return parsed


def extract_target_context_with_llm(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    llm_generate_fn: Callable[[str], str],
) -> str:
    """
    Backward-compatible LLM helper returning only target_context as a string.

    New project code should use extract_target_context_with_llm_judge because
    it validates JSON and returns context_type, references, confidence, and
    ambiguity notes.
    """
    resolved_references, _ = resolve_references(
        target_comment,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
    )
    weighted_context = compute_context_weights(
        target_comment,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
    )
    heuristic_context_type = classify_context_type(
        target_comment,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
        weighted_context=weighted_context,
        resolved_references=resolved_references,
        ambiguities=[],
    )
    heuristic_target_context = fallback_extract_target_context_with_heuristics(
        target_comment,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
        context_type=heuristic_context_type,
        resolved_references=resolved_references,
        weighted_context=weighted_context,
    )

    result = extract_target_context_with_llm_judge(
        target_comment=target_comment,
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        weighted_context=weighted_context,
        resolved_references=resolved_references,
        heuristic_context_type=heuristic_context_type,
        heuristic_target_context=heuristic_target_context,
        llm_generate_fn=llm_generate_fn,
    )
    return result.get("target_context", "") if result else ""


def extract_target_context(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    context_type: Optional[str] = None,
    resolved_references: Optional[Sequence[Mapping[str, Any]]] = None,
    weighted_context: Optional[Sequence[Mapping[str, Any]]] = None,
    llm_generate_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Infer the target-specific context behind the target comment.

    For full LLM-judge behavior, call analyze_target_comment/analyze_targets
    with llm_generate_fn. This function remains string-returning for backward
    compatibility and falls back to transparent heuristics whenever the LLM
    output is invalid or non-extractive.
    """
    if llm_generate_fn is not None:
        llm_result = extract_target_context_with_llm(
            target_comment=target_comment,
            post=post,
            parent=parent,
            ancestors=ancestors,
            prior_siblings=prior_siblings,
            prior_thread_comments=prior_thread_comments,
            llm_generate_fn=llm_generate_fn,
        )
        if llm_result:
            return llm_result

    return fallback_extract_target_context_with_heuristics(
        target_comment=target_comment,
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        context_type=context_type,
        resolved_references=resolved_references,
        weighted_context=weighted_context,
    )


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Deduplicate strings while preserving order."""
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output


def collect_ambiguities(
    target_comment: Mapping[str, Any],
    post: Optional[Mapping[str, Any]],
    parent: Optional[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
    prior_siblings: Sequence[Mapping[str, Any]],
    prior_thread_comments: Sequence[Mapping[str, Any]],
    resolved_reference_ambiguities: Sequence[str],
    resolved_references: Sequence[Mapping[str, Any]],
    context_type: str,
) -> List[str]:
    """Collect missing-context and uncertainty warnings."""
    ambiguities = list(resolved_reference_ambiguities)
    target_text = normalize_text(target_comment.get("_body", ""))

    if is_deleted_or_empty(target_text):
        ambiguities.append("The target comment is deleted or empty.")

    if len(token_set(target_text)) <= 2:
        ambiguities.append("The target comment is very short, so context inference is limited.")

    if resolved_references:
        low_confidence_refs = [
            ref for ref in resolved_references if float(ref.get("confidence", 0.0)) < 0.45
        ]
        if low_confidence_refs:
            ambiguities.append("One or more vague references could not be resolved confidently.")

    if parent is None and target_comment.get("_parent_id") not in {None, "", target_comment.get("_post_id")}:
        ambiguities.append("The parent comment is missing from the available comments table.")

    if post and not normalize_text(post.get("_body", "")):
        ambiguities.append("The post body is missing or empty.")

    if post is None:
        ambiguities.append("The post record for this target comment is missing.")

    if target_comment.get("_timestamp_order") is None and target_comment.get("_created_utc") is None:
        ambiguities.append("No timestamp_order or created_utc was available for the target; input row order was used.")

    if SARCASM_MARKER_RE.search(target_text):
        ambiguities.append("Sarcasm or irony may be present, which makes context inference less certain.")

    if context_type == "ambiguous":
        ambiguities.append("Multiple or insufficient prior contexts make the target context ambiguous.")

    if not prior_thread_comments and parent is not None:
        ambiguities.append("No earlier same-post comments were available besides local context, limiting broader-thread inference.")

    return dedupe_preserve_order(ambiguities)


def estimate_confidence(
    target_comment: Mapping[str, Any],
    context_type: str,
    weighted_context: Sequence[Mapping[str, Any]],
    resolved_references: Sequence[Mapping[str, Any]],
    ambiguities: Sequence[str],
) -> float:
    """Estimate confidence in the extracted context, not factual truth."""
    target_text = normalize_text(target_comment.get("_body", ""))

    if is_deleted_or_empty(target_text):
        return 0.05

    base_by_type = {
        "ambiguous": 0.35,
        "parent_reply": 0.72,
        "post_direct": 0.70,
        "ancestor_chain": 0.66,
        "prior_sibling_reference": 0.62,
        "broader_thread_reference": 0.55,
        "topic_shift": 0.58,
    }
    base = base_by_type.get(context_type, 0.45)

    top_context_weight = max(
        [float(item.get("relevance_weight", 0.0)) for item in weighted_context],
        default=0.0,
    )
    parent_context_weight = _top_weight_by_source(weighted_context, "direct_parent")
    post_context_weight = max(
        _top_weight_by_source(weighted_context, "post_title"),
        _top_weight_by_source(weighted_context, "post_body"),
    )

    base += 0.08 * top_context_weight
    base += 0.05 * max(parent_context_weight, post_context_weight)

    if resolved_references:
        avg_ref_confidence = sum(float(ref.get("confidence", 0.0)) for ref in resolved_references) / len(resolved_references)
        unresolved_count = sum(1 for ref in resolved_references if float(ref.get("confidence", 0.0)) < 0.45)
        base += 0.10 * avg_ref_confidence
        base -= 0.05 * unresolved_count

    informative_terms = len(token_set(target_text))
    if informative_terms <= 1:
        base -= 0.20
    elif informative_terms <= 3:
        base -= 0.10
    elif informative_terms >= 8:
        base += 0.05

    base -= min(0.25, 0.035 * len(ambiguities))
    return round(max(0.0, min(1.0, base)), 2)


def analyze_target_comment(
    target_comment_id: Any,
    posts: RecordsLike,
    comments: RecordsLike,
    llm_generate_fn: Optional[Callable[[str], str]] = None,
    use_llm_judge: bool = True,
    min_target_context_words: int = MIN_TARGET_CONTEXT_WORDS,
) -> Dict[str, Any]:
    """
    Analyze one target comment and return a Python dictionary.

    Deterministic reconstruction and heuristic extraction always run first.
    If use_llm_judge is True and llm_generate_fn is provided, the LLM may
    override the extracted context only when it returns valid JSON and
    target_context is extractive from the supplied prior text. The returned
    target_context is expanded to at least min_target_context_words whenever
    source context is available.
    """
    thread = prepare_thread_data(posts, comments)
    target_id = canonical_id(target_comment_id)

    if target_id is None or target_id not in thread.comment_index:
        return {
            "target_comment_id": target_comment_id,
            "post_context": "Post context unavailable.",
            "local_context": "Target comment not found.",
            "target_context": "The target comment was not found in the comments table.",
            "context_type": "ambiguous",
            "resolved_references": [],
            "weighted_context": [],
            "confidence": 0.0,
            "ambiguities": ["Target comment ID was not found."],
            "extraction_method": "heuristic_fallback",
        }

    target = thread.comment_index[target_id]
    post_id = target.get("_post_id")
    post = thread.post_index.get(post_id) if post_id is not None else None
    same_post_comments = thread.comments_by_post.get(post_id, [])

    parent = get_direct_parent(target, thread.comment_index)
    ancestors = get_ancestor_chain(target, thread.comment_index)
    prior_siblings = get_prior_siblings(target, same_post_comments)
    prior_thread_comments = get_prior_thread_comments(target, same_post_comments)

    post_context = extract_post_context(post)
    local_context = extract_local_context(target, parent, ancestors, prior_siblings)

    resolved_references, reference_ambiguities = resolve_references(
        target,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
    )

    weighted_context = compute_context_weights(
        target,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
    )

    context_type = classify_context_type(
        target,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
        weighted_context=weighted_context,
        resolved_references=resolved_references,
        ambiguities=reference_ambiguities,
    )

    target_context = fallback_extract_target_context_with_heuristics(
        target_comment=target,
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        context_type=context_type,
        resolved_references=resolved_references,
        weighted_context=weighted_context,
    )

    target_context = build_expanded_target_context(
        target_context=target_context,
        target_comment=target,
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        weighted_context=weighted_context,
        context_type=context_type,
        min_words=min_target_context_words,
    )

    ambiguities = collect_ambiguities(
        target,
        post,
        parent,
        ancestors,
        prior_siblings,
        prior_thread_comments,
        reference_ambiguities,
        resolved_references,
        context_type,
    )

    confidence = estimate_confidence(
        target,
        context_type,
        weighted_context,
        resolved_references,
        ambiguities,
    )

    heuristic_result: Dict[str, Any] = {
        "target_comment_id": target.get("_comment_id", target_comment_id),
        "post_context": post_context,
        "local_context": local_context,
        "target_context": target_context,
        "context_type": context_type,
        "resolved_references": resolved_references,
        "weighted_context": weighted_context,
        "confidence": confidence,
        "ambiguities": ambiguities,
        "extraction_method": "heuristic_fallback",
    }

    if not use_llm_judge or llm_generate_fn is None:
        return heuristic_result

    llm_result = extract_target_context_with_llm_judge(
        target_comment=target,
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        weighted_context=weighted_context,
        resolved_references=resolved_references,
        heuristic_context_type=context_type,
        heuristic_target_context=target_context,
        llm_generate_fn=llm_generate_fn,
    )

    if llm_result is None:
        return heuristic_result

    structural_ambiguities = [
        note
        for note in ambiguities
        if any(
            marker in note.lower()
            for marker in ("missing", "unavailable", "deleted", "empty", "timestamp")
        )
    ]

    expanded_llm_target_context = build_expanded_target_context(
        target_context=llm_result["target_context"],
        target_comment=target,
        post=post,
        parent=parent,
        ancestors=ancestors,
        prior_siblings=prior_siblings,
        prior_thread_comments=prior_thread_comments,
        weighted_context=weighted_context,
        context_type=llm_result["context_type"],
        min_words=min_target_context_words,
    )

    return {
        "target_comment_id": target.get("_comment_id", target_comment_id),
        "post_context": post_context,
        "local_context": local_context,
        "target_context": expanded_llm_target_context,
        "context_type": llm_result["context_type"],
        "resolved_references": llm_result["resolved_references"],
        "weighted_context": weighted_context,
        "confidence": llm_result["confidence"],
        "ambiguities": dedupe_preserve_order(
            list(llm_result.get("ambiguities", [])) + structural_ambiguities
        ),
        "extraction_method": "llm_judge",
    }


def analyze_targets(
    posts: RecordsLike,
    comments: RecordsLike,
    target_comment_ids: Sequence[Any],
    llm_generate_fn: Optional[Callable[[str], str]] = None,
    use_llm_judge: bool = True,
    min_target_context_words: int = MIN_TARGET_CONTEXT_WORDS,
) -> Dict[str, Any]:
    """
    Analyze multiple target comments and return Python data structures.

    Set use_llm_judge=False for deterministic heuristic-only extraction.
    """
    thread = prepare_thread_data(posts, comments)
    results: List[Dict[str, Any]] = []
    post_ids: List[str] = []

    for target_comment_id in target_comment_ids:
        target_id = canonical_id(target_comment_id)
        target = thread.comment_index.get(target_id) if target_id is not None else None
        if target and target.get("_post_id") is not None:
            post_ids.append(str(target.get("_post_id")))

        results.append(
            analyze_target_comment(
                target_comment_id=target_comment_id,
                posts=thread.posts,
                comments=thread.comments,
                llm_generate_fn=llm_generate_fn,
                use_llm_judge=use_llm_judge,
                min_target_context_words=min_target_context_words,
            )
        )

    unique_post_ids = dedupe_preserve_order(post_ids)
    output: Dict[str, Any] = {
        "post_id": unique_post_ids[0] if len(unique_post_ids) == 1 else None,
        "results": results,
    }

    if len(unique_post_ids) > 1:
        output["post_ids"] = unique_post_ids

    return output


def print_result_compact(results: Mapping[str, Any]) -> None:
    """Small helper for the example usage block."""
    print("post_id:", results.get("post_id"))
    for item in results.get("results", []):
        print()
        print("target_comment_id:", item.get("target_comment_id"))
        print("context_type:", item.get("context_type"))
        print("target_context:", item.get("target_context"))
        print("confidence:", item.get("confidence"))
        print("extraction_method:", item.get("extraction_method"))
        if item.get("resolved_references"):
            print("resolved_references:", item.get("resolved_references"))
        if item.get("ambiguities"):
            print("ambiguities:", item.get("ambiguities"))


if __name__ == "__main__":
    mock_posts = [
        {
            "post_id": "p1",
            "title": "City council approves a new bike lane plan",
            "body": (
                "The plan adds protected bike lanes downtown and removes some street parking. "
                "The article says construction may start in June."
            ),
            "subreddit": "examplecity",
            "author": "op_user",
            "created_utc": 1000,
        }
    ]

    mock_comments = [
        {
            "comment_id": "c1",
            "post_id": "p1",
            "parent_id": "p1",
            "body": "Removing parking is going to hurt small businesses on that street.",
            "author": "user_a",
            "created_utc": 1010,
        },
        {
            "comment_id": "c2",
            "post_id": "p1",
            "parent_id": "p1",
            "body": "The article mentions the city surveyed nearby shops first.",
            "author": "user_b",
            "created_utc": 1020,
        },
        {
            "comment_id": "c3",
            "post_id": "p1",
            "parent_id": "c1",
            "body": "Do we know if those businesses actually rely on curb parking?",
            "author": "user_c",
            "created_utc": 1030,
        },
        {
            "comment_id": "c4",
            "post_id": "p1",
            "parent_id": "c3",
            "body": "That is exactly what I was wondering too.",
            "author": "user_d",
            "created_utc": 1040,
        },
        {
            "comment_id": "c5",
            "post_id": "p1",
            "parent_id": "p1",
            "body": "What about delivery trucks?",
            "author": "user_e",
            "created_utc": 1050,
        },
    ]

    heuristic_only_results = analyze_targets(
        posts=mock_posts,
        comments=mock_comments,
        target_comment_ids=["c4", "c5"],
        llm_generate_fn=None,
        use_llm_judge=False,
        min_target_context_words=MIN_TARGET_CONTEXT_WORDS,
    )
    print_result_compact(heuristic_only_results)

    # Minimal local OLMo-style usage.
    # Replace the body of local_olmo_generate with tokenizer/model.generate.
    # The required interface is: llm_generate_fn(prompt: str) -> str.
    def local_olmo_generate(prompt: str) -> str:
        """
        Example injectable local generation function.

        This placeholder returns an empty string, so the script uses the
        heuristic fallback. Replace it with a local OLMo inference call.
        """
        return ""

    # llm_judge_results = analyze_targets(
    #     posts=mock_posts,
    #     comments=mock_comments,
    #     target_comment_ids=["c4", "c5"],
    #     llm_generate_fn=local_olmo_generate,
    #     use_llm_judge=True,
    #     min_target_context_words=MIN_TARGET_CONTEXT_WORDS,
    # )
    # print_result_compact(llm_judge_results)
