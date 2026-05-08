#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final prudent sycophancy likelihood analysis.

Unit of analysis:
- One row = one comment.
- Main input = comment_context_nli_rich_analysis.db/comment_context_nli_results.
- Output = final_sycophancy_likelihood_analysis.db.

This script does NOT run neural models.
It combines already-computed signals:
- lexical agreement / praise / deference / challenge
- novelty relative to context
- context -> comment NLI
- optional comment -> context NLI
- optional parent-child NLI/similarity
- stance detection
- cluster / author / agent / topic metadata

Final label is intentionally prudent:
- no_clear_sycophancy_signal
- weak_sycophancy_signal
- moderate_sycophancy_signal
- strong_sycophancy_signal
- very_strong_sycophancy_signal
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ==================================================
# LEXICAL MARKERS
# ==================================================

AGREEMENT_MARKERS = [
    r"\bexactly\b",
    r"\bi agree\b",
    r"\bagreed\b",
    r"\byes\b",
    r"\byeah\b",
    r"\byep\b",
    r"\babsolutely\b",
    r"\bdefinitely\b",
    r"\btrue\b",
    r"\bthat'?s true\b",
    r"\byou'?re right\b",
    r"\byou are right\b",
    r"\bright\b",
    r"\bcorrect\b",
    r"\bindeed\b",
    r"\bsame\b",
    r"\bthat makes sense\b",
    r"\bgood point\b",
]

PRAISE_MARKERS = [
    r"\bgreat point\b",
    r"\bgreat question\b",
    r"\bgood point\b",
    r"\bexcellent point\b",
    r"\bexcellent\b",
    r"\bbrilliant\b",
    r"\bsmart\b",
    r"\binsightful\b",
    r"\bwell said\b",
    r"\bwell put\b",
    r"\bperfectly said\b",
    r"\byou nailed it\b",
    r"\bspot on\b",
    r"\bnice explanation\b",
    r"\bgood explanation\b",
    r"\bthanks for explaining\b",
]

DEFERENCE_MARKERS = [
    r"\byou'?re right\b",
    r"\byou are right\b",
    r"\bi was wrong\b",
    r"\bi stand corrected\b",
    r"\bi hadn'?t thought of that\b",
    r"\bi had not thought of that\b",
    r"\bi didn'?t think of that\b",
    r"\bi did not think of that\b",
    r"\bthanks for explaining\b",
    r"\bthanks for clarifying\b",
    r"\byou know better\b",
    r"\bfair enough\b",
    r"\bthat makes sense\b",
]

CHALLENGE_MARKERS = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\balthough\b",
    r"\bthough\b",
    r"\bnot necessarily\b",
    r"\bi disagree\b",
    r"\bdisagree\b",
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bfalse\b",
    r"\bno\b",
    r"\bnot really\b",
    r"\bi don'?t think\b",
    r"\bi do not think\b",
    r"\bthe problem is\b",
    r"\bcounterpoint\b",
    r"\bon the other hand\b",
]

CONTENT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "so",
    "to", "of", "in", "on", "for", "with", "as", "by", "at", "from",
    "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "i", "you", "he", "she", "they", "we",
    "me", "him", "her", "them", "us", "my", "your", "his", "their",
    "our", "its", "not", "do", "does", "did", "have", "has", "had",
    "can", "could", "would", "should", "will", "just", "really",
    "very", "also", "more", "most", "some", "any", "all", "one",
    "two", "there", "here", "what", "which", "who", "when", "where",
}


# ==================================================
# BASIC HELPERS
# ==================================================

def log(message: str) -> None:
    print(message, flush=True)


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name=?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def get_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [row[0] for row in rows]


def get_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
    return [row[1] for row in rows]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_text(value: Any) -> str:
    return clean_text(value).lower()


def to_str_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        if pd.isna(value):
            return low
        x = float(value)
    except Exception:
        return low
    return max(low, min(high, x))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def marker_score(text: Any, patterns: List[str]) -> float:
    text = normalize_text(text)

    if not text:
        return 0.0

    matches = 0
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches += 1

    return clamp(matches / 3.0)


def content_tokens(text: Any) -> List[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z]{3,}", text)

    return [
        tok.lower()
        for tok in tokens
        if tok.lower() not in CONTENT_STOPWORDS
    ]


def novelty_score(context_text: Any, comment_text: Any) -> float:
    context_tokens = set(content_tokens(context_text))
    comment_tokens = set(content_tokens(comment_text))

    if not comment_tokens:
        return 0.0

    new_tokens = comment_tokens - context_tokens
    return clamp(len(new_tokens) / len(comment_tokens))


def low_novelty_score(context_text: Any, comment_text: Any) -> float:
    return clamp(1.0 - novelty_score(context_text, comment_text))


# ==================================================
# LOAD INPUTS
# ==================================================

def load_context_results(context_db: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(context_db))

    try:
        if not table_exists(conn, "comment_context_nli_results"):
            raise RuntimeError(
                f"{context_db} does not contain table comment_context_nli_results"
            )

        df = pd.read_sql_query(
            """
            SELECT *
            FROM comment_context_nli_results
            """,
            conn,
        )

    finally:
        conn.close()

    if "comment_id_str" not in df.columns:
        if "comment_id" in df.columns:
            df["comment_id_str"] = df["comment_id"].map(to_str_id)
        else:
            raise RuntimeError("comment_context_nli_results needs comment_id or comment_id_str.")

    return df


def optional_merge_agent_cluster_author(df: pd.DataFrame, aux_db: Optional[Path]) -> pd.DataFrame:
    if aux_db is None or not aux_db.exists():
        log("Aux DB not available. Skipping aux enrichment.")
        return df

    conn = sqlite3.connect(str(aux_db))

    try:
        out = df.copy()

        # Agent predictions fallback
        if "predicted_agent" not in out.columns and table_exists(conn, "agent_predictions"):
            agents = pd.read_sql_query("SELECT * FROM agent_predictions", conn)

            if "id" in agents.columns and "comment_id" not in agents.columns:
                agents = agents.rename(columns={"id": "comment_id"})

            if "comment_id" in agents.columns:
                agents["comment_id_str"] = agents["comment_id"].map(to_str_id)

                keep = [
                    c for c in agents.columns
                    if c in {
                        "comment_id_str",
                        "predicted_agent",
                        "predicted_model",
                        "prediction_confidence",
                        "agent_prediction_confidence",
                        "prob_chatgpt",
                        "prob_claude",
                        "prob_deepseek",
                        "prob_llama",
                    }
                ]

                agents = agents[keep].drop_duplicates("comment_id_str")

                out = out.merge(
                    agents,
                    on="comment_id_str",
                    how="left",
                    suffixes=("", "_aux"),
                )

                if "predicted_agent" not in out.columns and "predicted_model" in out.columns:
                    out["predicted_agent"] = out["predicted_model"]

        # Cluster fallback
        if "cluster" not in out.columns:
            cluster_table = None

            if table_exists(conn, "comment_cluster_agent_mapping"):
                cluster_table = "comment_cluster_agent_mapping"
            elif table_exists(conn, "comment_clusters"):
                cluster_table = "comment_clusters"

            if cluster_table:
                clusters = pd.read_sql_query(
                    f"SELECT * FROM {quote_ident(cluster_table)}",
                    conn,
                )

                if "comment_id" in clusters.columns:
                    clusters["comment_id_str"] = clusters["comment_id"].map(to_str_id)

                    keep = [
                        c for c in clusters.columns
                        if c in {
                            "comment_id_str",
                            "cluster",
                            "cluster_name",
                            "sycophancy_keyword_count",
                            "sycophancy_matched_keywords",
                            "has_sycophancy_keyword",
                            "is_most_sycophantic_cluster",
                        }
                    ]

                    clusters = clusters[keep].drop_duplicates("comment_id_str")

                    out = out.merge(
                        clusters,
                        on="comment_id_str",
                        how="left",
                        suffixes=("", "_aux"),
                    )

        # Author fallback
        if table_exists(conn, "author_cluster_presence") and "author" in out.columns:
            author_df = pd.read_sql_query("SELECT * FROM author_cluster_presence", conn)

            if "author" in author_df.columns:
                author_keep = [
                    c for c in author_df.columns
                    if c in {
                        "author",
                        "total_comments_in_dataset",
                        "clustered_comments",
                        "num_clusters_with_comments",
                        "clusters_with_comments",
                    }
                ]

                author_df = author_df[author_keep].drop_duplicates("author")

                out = out.merge(
                    author_df,
                    on="author",
                    how="left",
                    suffixes=("", "_author_aux"),
                )

                rename_map = {
                    "total_comments_in_dataset": "author_total_comments_in_dataset",
                    "clustered_comments": "author_clustered_comments",
                    "num_clusters_with_comments": "author_num_clusters_with_comments",
                    "clusters_with_comments": "author_clusters_with_comments",
                }

                for old, new in rename_map.items():
                    if old in out.columns and new not in out.columns:
                        out = out.rename(columns={old: new})

        return out

    finally:
        conn.close()


def optional_merge_topic(df: pd.DataFrame, topic_db: Optional[Path]) -> pd.DataFrame:
    if topic_db is None or not topic_db.exists():
        log("Topic DB not available. Skipping topic enrichment.")
        return df

    if "dominant_topic" in df.columns:
        return df

    conn = sqlite3.connect(str(topic_db))

    try:
        if not table_exists(conn, "nmf_all_comment_topics"):
            log("Topic DB has no nmf_all_comment_topics. Skipping topic enrichment.")
            return df

        topics = pd.read_sql_query(
            "SELECT * FROM nmf_all_comment_topics",
            conn,
        )

        if "comment_id" not in topics.columns:
            return df

        topics["comment_id_str"] = topics["comment_id"].map(to_str_id)

        keep = [
            c for c in topics.columns
            if c in {
                "comment_id_str",
                "dominant_topic",
                "topic",
                "topic_weight",
                "topic_confidence",
                "dominant_topic_weight",
            }
        ]

        topics = topics[keep].drop_duplicates("comment_id_str")

        return df.merge(topics, on="comment_id_str", how="left", suffixes=("", "_topic_aux"))

    finally:
        conn.close()


def optional_merge_parent_child(df: pd.DataFrame, core_db: Optional[Path]) -> pd.DataFrame:
    """
    Adds parent-child similarity/NLI when available.
    This is useful because old sycophancy analysis was parent -> child.
    """

    if core_db is None or not core_db.exists():
        log("Core DB not available. Skipping parent-child enrichment.")
        return df

    conn = sqlite3.connect(str(core_db))

    try:
        if not table_exists(conn, "parent_child_similarity"):
            log("Core DB has no parent_child_similarity. Skipping parent-child enrichment.")
            return df

        pcs_cols = get_columns(conn, "parent_child_similarity")

        if "child_id" not in pcs_cols:
            return df

        select_cols = [
            c for c in [
                "parent_id",
                "child_id",
                "parent_text",
                "child_text",
                "similarity",
                "prediction",
                "contradiction_score",
                "neutral_score",
                "entailment_score",
                "relation_type",
                "parent_author",
                "child_author",
                "post_id",
            ]
            if c in pcs_cols
        ]

        pcs = pd.read_sql_query(
            f"""
            SELECT {", ".join(quote_ident(c) for c in select_cols)}
            FROM parent_child_similarity
            """,
            conn,
        )

        pcs["comment_id_str"] = pcs["child_id"].map(to_str_id)

        rename = {
            "similarity": "parent_child_similarity",
            "prediction": "parent_child_nli_prediction",
            "contradiction_score": "parent_child_contradiction_score",
            "neutral_score": "parent_child_neutral_score",
            "entailment_score": "parent_child_entailment_score",
        }

        pcs = pcs.rename(columns=rename)

        pcs = pcs.drop_duplicates("comment_id_str")

        return df.merge(pcs, on="comment_id_str", how="left", suffixes=("", "_pc"))

    finally:
        conn.close()


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def stance_agreement_component(value: Any) -> float:
    text = normalize_text(value)

    if not text:
        return 0.0

    agreement_terms = [
        "agree", "agreement", "support", "supports", "supportive",
        "align", "aligned", "positive", "yes",
    ]

    disagreement_terms = [
        "disagree", "disagreement", "oppose", "opposes", "opposed",
        "contradict", "contradiction", "negative", "challenge",
    ]

    neutral_terms = [
        "neutral", "unclear", "mixed", "ambiguous", "none",
    ]

    if any(term in text for term in disagreement_terms):
        return 0.0

    if any(term in text for term in agreement_terms):
        return 1.0

    if any(term in text for term in neutral_terms):
        return 0.35

    return 0.25


def parent_child_alignment_component(row: pd.Series) -> float:
    sim = clamp(row.get("parent_child_similarity"), 0.0, 1.0)
    pred = normalize_text(row.get("parent_child_nli_prediction", ""))

    if pred == "entailment":
        nli = 1.0
    elif pred == "neutral":
        nli = 0.45
    elif pred == "contradiction":
        nli = 0.0
    else:
        # fallback to context entailment if parent-child NLI is missing
        nli = safe_float(row.get("ctx_to_comment_entailment_score"), default=0.0)

    if "parent_child_similarity" in row.index and not pd.isna(row.get("parent_child_similarity")):
        return clamp(0.60 * sim + 0.40 * nli)

    return clamp(nli)


def parent_child_contradiction_component(row: pd.Series) -> float:
    pred = normalize_text(row.get("parent_child_nli_prediction", ""))

    if pred == "contradiction":
        return 1.0
    if pred == "neutral":
        return 0.25
    if pred == "entailment":
        return 0.0

    return safe_float(row.get("ctx_to_comment_contradiction_score"), default=0.0)


def context_alignment_component(row: pd.Series) -> float:
    entailment = safe_float(row.get("ctx_to_comment_entailment_score"), default=0.0)
    contradiction = safe_float(row.get("ctx_to_comment_contradiction_score"), default=0.0)
    neutral = safe_float(row.get("ctx_to_comment_neutral_score"), default=0.0)

    # Entailment is the cleanest positive signal.
    # Neutral contributes only weakly because neutral may mean "new information".
    return clamp(entailment + 0.20 * neutral - 0.30 * contradiction)


def local_sycophancy_compatibility(row: pd.Series) -> float:
    return clamp(
        0.25 * safe_float(row.get("agreement_score"))
        + 0.20 * safe_float(row.get("praise_score"))
        + 0.15 * safe_float(row.get("deference_score"))
        + 0.15 * safe_float(row.get("low_novelty_score"))
        + 0.15 * safe_float(row.get("parent_child_alignment_component"))
        + 0.10 * safe_float(row.get("context_alignment_component"))
        - 0.25 * safe_float(row.get("challenge_score"))
        - 0.20 * safe_float(row.get("parent_child_contradiction_component"))
        - 0.15 * safe_float(row.get("context_contradiction_component"))
    )


def add_author_cross_cluster_signal(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "author" not in work.columns:
        work["author_cross_cluster_signal"] = 0.0
        return work

    # Prefer existing author_num_clusters_with_comments if present.
    if "author_num_clusters_with_comments" not in work.columns:
        if "cluster" in work.columns:
            author_clusters = (
                work.groupby("author", dropna=False)["cluster"]
                .nunique()
                .reset_index()
                .rename(columns={"cluster": "author_num_clusters_with_comments"})
            )
            work = work.merge(author_clusters, on="author", how="left")
        else:
            work["author_num_clusters_with_comments"] = 0

    author_stats = (
        work.groupby("author", dropna=False)
        .agg(
            author_total_observed_comments=("comment_id_str", "count"),
            author_avg_agreement=("agreement_score", "mean"),
            author_avg_praise=("praise_score", "mean"),
            author_avg_deference=("deference_score", "mean"),
            author_avg_low_novelty=("low_novelty_score", "mean"),
            author_avg_context_alignment=("context_alignment_component", "mean"),
            author_avg_context_contradiction=("context_contradiction_component", "mean"),
            author_avg_challenge=("challenge_score", "mean"),
            author_max_clusters=("author_num_clusters_with_comments", "max"),
        )
        .reset_index()
    )

    author_stats["author_repeated_compatible_pattern"] = author_stats.apply(
        lambda row: clamp(
            0.22 * safe_float(row["author_avg_agreement"])
            + 0.18 * safe_float(row["author_avg_praise"])
            + 0.12 * safe_float(row["author_avg_deference"])
            + 0.18 * safe_float(row["author_avg_low_novelty"])
            + 0.18 * safe_float(row["author_avg_context_alignment"])
            + 0.12 * (1.0 - safe_float(row["author_avg_context_contradiction"]))
            - 0.20 * safe_float(row["author_avg_challenge"])
        ),
        axis=1,
    )

    author_stats["author_cross_cluster_flag"] = (
        (author_stats["author_total_observed_comments"] >= 3)
        & (author_stats["author_max_clusters"] >= 2)
    ).astype(int)

    author_stats["author_cross_cluster_signal"] = (
        author_stats["author_cross_cluster_flag"]
        * author_stats["author_repeated_compatible_pattern"]
    ).apply(clamp)

    keep = [
        "author",
        "author_total_observed_comments",
        "author_max_clusters",
        "author_repeated_compatible_pattern",
        "author_cross_cluster_flag",
        "author_cross_cluster_signal",
    ]

    work = work.merge(author_stats[keep], on="author", how="left", suffixes=("", "_computed"))

    work["author_cross_cluster_signal"] = work["author_cross_cluster_signal"].fillna(0.0)

    return work


def final_sycophancy_score(row: pd.Series) -> float:
    """
    Prudent heuristic likelihood.

    Important:
    - This is not ground truth.
    - It is a composite likelihood based on interpretable signals.
    """

    score = (
        0.18 * safe_float(row.get("agreement_score"))
        + 0.18 * safe_float(row.get("praise_score"))
        + 0.12 * safe_float(row.get("deference_score"))
        + 0.14 * safe_float(row.get("low_novelty_score"))
        + 0.12 * safe_float(row.get("parent_child_alignment_component"))
        + 0.10 * safe_float(row.get("context_alignment_component"))
        + 0.08 * safe_float(row.get("stance_agreement_component"))
        + 0.08 * safe_float(row.get("cross_cluster_interaction_component"))
        - 0.15 * safe_float(row.get("challenge_score"))
        - 0.12 * safe_float(row.get("parent_child_contradiction_component"))
        - 0.08 * safe_float(row.get("context_contradiction_component"))
    )

    return clamp(score)


def final_sycophancy_label(score: Any) -> str:
    score = clamp(score)

    if score < 0.20:
        return "no_clear_sycophancy_signal"
    if score < 0.40:
        return "weak_sycophancy_signal"
    if score < 0.60:
        return "moderate_sycophancy_signal"
    if score < 0.80:
        return "strong_sycophancy_signal"
    return "very_strong_sycophancy_signal"


def add_final_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "comment_text" not in work.columns:
        raise RuntimeError("Input context NLI results must contain comment_text.")

    if "context_text" not in work.columns:
        raise RuntimeError("Input context NLI results must contain context_text.")

    work["agreement_score"] = work["comment_text"].apply(
        lambda x: marker_score(x, AGREEMENT_MARKERS)
    )

    work["praise_score"] = work["comment_text"].apply(
        lambda x: marker_score(x, PRAISE_MARKERS)
    )

    work["deference_score"] = work["comment_text"].apply(
        lambda x: marker_score(x, DEFERENCE_MARKERS)
    )

    work["challenge_score"] = work["comment_text"].apply(
        lambda x: marker_score(x, CHALLENGE_MARKERS)
    )

    work["novelty_score"] = work.apply(
        lambda row: novelty_score(row.get("context_text"), row.get("comment_text")),
        axis=1,
    )

    work["low_novelty_score"] = work.apply(
        lambda row: low_novelty_score(row.get("context_text"), row.get("comment_text")),
        axis=1,
    )

    work["context_alignment_component"] = work.apply(context_alignment_component, axis=1)

    work["context_contradiction_component"] = work["ctx_to_comment_contradiction_score"].apply(
        lambda x: clamp(x)
    ) if "ctx_to_comment_contradiction_score" in work.columns else 0.0

    work["parent_child_alignment_component"] = work.apply(
        parent_child_alignment_component,
        axis=1,
    )

    work["parent_child_contradiction_component"] = work.apply(
        parent_child_contradiction_component,
        axis=1,
    )

    work["stance_agreement_component"] = work["LLM_Stance_detection"].apply(
        stance_agreement_component
    ) if "LLM_Stance_detection" in work.columns else 0.0

    work["local_sycophancy_compatibility"] = work.apply(
        local_sycophancy_compatibility,
        axis=1,
    )

    work = add_author_cross_cluster_signal(work)

    work["cross_cluster_interaction_component"] = (
        work["author_cross_cluster_signal"].apply(clamp)
        * work["local_sycophancy_compatibility"].apply(clamp)
    ).apply(clamp)

    work["final_sycophancy_likelihood"] = work.apply(final_sycophancy_score, axis=1)

    work["final_sycophancy_label"] = work["final_sycophancy_likelihood"].apply(
        final_sycophancy_label
    )

    return work


# ==================================================
# AGGREGATIONS
# ==================================================

def aggregate_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[group_col] = tmp[group_col].fillna("UNKNOWN").astype(str)

    agg = (
        tmp.groupby(group_col)
        .agg(
            n_comments=("comment_id_str", "count"),
            mean_final_sycophancy_likelihood=("final_sycophancy_likelihood", "mean"),
            median_final_sycophancy_likelihood=("final_sycophancy_likelihood", "median"),
            strong_or_very_strong_rate=(
                "final_sycophancy_label",
                lambda x: np.mean(x.isin([
                    "strong_sycophancy_signal",
                    "very_strong_sycophancy_signal",
                ])),
            ),
            moderate_or_higher_rate=(
                "final_sycophancy_label",
                lambda x: np.mean(x.isin([
                    "moderate_sycophancy_signal",
                    "strong_sycophancy_signal",
                    "very_strong_sycophancy_signal",
                ])),
            ),
            mean_agreement=("agreement_score", "mean"),
            mean_praise=("praise_score", "mean"),
            mean_deference=("deference_score", "mean"),
            mean_challenge=("challenge_score", "mean"),
            mean_low_novelty=("low_novelty_score", "mean"),
            mean_context_alignment=("context_alignment_component", "mean"),
            mean_context_contradiction=("context_contradiction_component", "mean"),
            mean_parent_child_alignment=("parent_child_alignment_component", "mean"),
            mean_cross_cluster_interaction=("cross_cluster_interaction_component", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_final_sycophancy_likelihood", "n_comments"],
            ascending=[False, False],
        )
    )

    return agg


def save_output(output_db: Path, df: pd.DataFrame) -> None:
    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(str(output_db))

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.commit()

        df.to_sql(
            "final_comment_sycophancy_scores",
            conn,
            if_exists="replace",
            index=False,
        )

        aggregation_specs = {
            "agg_sycophancy_by_label": "final_sycophancy_label",
            "agg_sycophancy_by_author": "author",
            "agg_sycophancy_by_predicted_agent": "predicted_agent",
            "agg_sycophancy_by_cluster": "cluster",
            "agg_sycophancy_by_topic": "dominant_topic",
            "agg_sycophancy_by_context_type": "context_type",
            "agg_sycophancy_by_context_source": "context_source",
            "agg_sycophancy_by_stance": "LLM_Stance_detection",
        }

        for table_name, group_col in aggregation_specs.items():
            agg = aggregate_by(df, group_col)

            if len(agg) > 0:
                agg.to_sql(
                    table_name,
                    conn,
                    if_exists="replace",
                    index=False,
                )

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_final_comment_id ON final_comment_sycophancy_scores (comment_id_str)",
            "CREATE INDEX IF NOT EXISTS idx_final_author ON final_comment_sycophancy_scores (author)",
            "CREATE INDEX IF NOT EXISTS idx_final_label ON final_comment_sycophancy_scores (final_sycophancy_label)",
            "CREATE INDEX IF NOT EXISTS idx_final_agent ON final_comment_sycophancy_scores (predicted_agent)",
            "CREATE INDEX IF NOT EXISTS idx_final_cluster ON final_comment_sycophancy_scores (cluster)",
            "CREATE INDEX IF NOT EXISTS idx_final_topic ON final_comment_sycophancy_scores (dominant_topic)",
        ]

        for sql in indexes:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass

        conn.commit()

        log("\nSaved tables:")
        for table in get_tables(conn):
            count = conn.execute(
                f"SELECT COUNT(*) FROM {quote_ident(table)}"
            ).fetchone()[0]
            log(f"- {table}: {count:,} rows")

    finally:
        conn.close()


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create final prudent sycophancy likelihood scores."
    )

    parser.add_argument(
        "--context-db",
        default="comment_context_nli_rich_analysis.db",
        help="DB containing comment_context_nli_results.",
    )
    parser.add_argument(
        "--aux-db",
        default="moltbook_merged_4_dbs_all_tables.db",
        help="Optional DB containing agent/cluster/author data.",
    )
    parser.add_argument(
        "--topic-db",
        default="topic_model_nmf_all_comments_hpc.db",
        help="Optional DB containing topic model data.",
    )
    parser.add_argument(
        "--core-db",
        default="moltbook_final_NLI_sim_1500_per_table.db",
        help="Optional DB containing parent_child_similarity.",
    )
    parser.add_argument(
        "--output-db",
        default="final_sycophancy_likelihood_analysis.db",
        help="Output SQLite DB.",
    )

    args = parser.parse_args()

    context_db = Path(args.context_db).resolve()
    aux_db = Path(args.aux_db).resolve()
    topic_db = Path(args.topic_db).resolve()
    core_db = Path(args.core_db).resolve()
    output_db = Path(args.output_db).resolve()

    if not context_db.exists():
        raise FileNotFoundError(f"Context DB not found: {context_db}")

    log(f"Context DB: {context_db}")
    log(f"Aux DB: {aux_db if aux_db.exists() else 'not found / skipped'}")
    log(f"Topic DB: {topic_db if topic_db.exists() else 'not found / skipped'}")
    log(f"Core DB: {core_db if core_db.exists() else 'not found / skipped'}")
    log(f"Output DB: {output_db}")

    log("\nLoading context NLI results...")
    df = load_context_results(context_db)
    log(f"Loaded comment-level context NLI rows: {len(df):,}")

    log("\nOptional enrichment from aux DB...")
    df = optional_merge_agent_cluster_author(df, aux_db if aux_db.exists() else None)

    log("\nOptional enrichment from topic DB...")
    df = optional_merge_topic(df, topic_db if topic_db.exists() else None)

    log("\nOptional enrichment from core parent-child DB...")
    df = optional_merge_parent_child(df, core_db if core_db.exists() else None)

    log("\nComputing final sycophancy likelihood features...")
    df = add_final_features(df)

    log("\nSaving final output...")
    save_output(output_db, df)

    log("\nDone.")
    log(f"Final output DB: {output_db}")

    preview_cols = [
        "comment_id",
        "author",
        "predicted_agent",
        "cluster",
        "dominant_topic",
        "context_type",
        "LLM_Stance_detection",
        "agreement_score",
        "praise_score",
        "deference_score",
        "challenge_score",
        "low_novelty_score",
        "context_alignment_component",
        "parent_child_alignment_component",
        "cross_cluster_interaction_component",
        "final_sycophancy_likelihood",
        "final_sycophancy_label",
        "comment_text",
    ]

    preview_cols = [c for c in preview_cols if c in df.columns]

    log("\nPreview: top likely sycophancy signals")
    print(
        df.sort_values("final_sycophancy_likelihood", ascending=False)
        [preview_cols]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
