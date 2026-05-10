"""
Rich NLI analysis: context -> comment.

Unit of analysis:
- One row = one comment with one selected best context.
- NLI pair = selected_context as premise, comment body as hypothesis.

Inputs:
- core DB: comments/posts/parent_child_similarity/context extraction
- aux DB: agent predictions, clusters, sycophancy, author-level info
- topic DB: NMF topic assignments per comment

Output:
- one SQLite DB with comment-level NLI results and aggregate tables.
"""

import argparse
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def is_empty_text(value: Any) -> bool:
    text = clean_text(value).lower()
    return text in {"", "nan", "none", "[deleted]", "[removed]", "deleted", "removed"}


def truncate_text(value: Any, max_chars: int) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + " ... [truncated]"


def to_str_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def confidence_bucket(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if x >= 0.80:
        return "high_0.80_1.00"
    if x >= 0.50:
        return "medium_0.50_0.79"
    if x >= 0.00:
        return "low_0.00_0.49"
    return "unknown"


def choose_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==================================================
# LOAD CORE DATA
# ==================================================

def load_comments(core_conn: sqlite3.Connection, limit: Optional[int]) -> pd.DataFrame:
    if not table_exists(core_conn, "comments"):
        raise RuntimeError("Core DB does not contain table 'comments'.")

    cols = get_columns(core_conn, "comments")

    if "id" not in cols or "body" not in cols:
        raise RuntimeError("comments table must contain at least 'id' and 'body'.")

    preferred_cols = [
        "id",
        "post_id",
        "parent_comment_id",
        "depth",
        "comment_order",
        "author",
        "body",
        "target_context",
        "context_type",
        "context_confidence",
        "context_ambiguities",
        "resolved_references",
        "weighted_context",
        "extraction_method",
        "Final_context_extracted",
        "LLM_Stance_detection",
        "LLM_Stance_detection_explanation",
    ]

    selected_cols = [c for c in preferred_cols if c in cols]

    sql = f"""
    SELECT {", ".join(quote_ident(c) for c in selected_cols)}
    FROM comments
    WHERE body IS NOT NULL
      AND TRIM(CAST(body AS TEXT)) != ''
    """

    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"

    df = pd.read_sql_query(sql, core_conn)

    df["comment_id_str"] = df["id"].map(to_str_id)

    if "post_id" in df.columns:
        df["post_id_str"] = df["post_id"].map(to_str_id)
    else:
        df["post_id_str"] = ""

    return df


def load_parent_context_map(core_conn: sqlite3.Connection) -> Dict[str, str]:
    """
    Maps child comment id -> parent_text from parent_child_similarity.
    Used only as fallback context.
    """

    if not table_exists(core_conn, "parent_child_similarity"):
        return {}

    cols = get_columns(core_conn, "parent_child_similarity")
    if "child_id" not in cols or "parent_text" not in cols:
        return {}

    df = pd.read_sql_query(
        """
        SELECT child_id, parent_text
        FROM parent_child_similarity
        WHERE child_id IS NOT NULL
          AND parent_text IS NOT NULL
          AND TRIM(CAST(parent_text AS TEXT)) != ''
        """,
        core_conn,
    )

    df["child_id_str"] = df["child_id"].map(to_str_id)

    out = {}
    for _, row in df.iterrows():
        text = clean_text(row["parent_text"])
        if not is_empty_text(text):
            out[row["child_id_str"]] = text

    return out


def load_post_context_map(core_conn: sqlite3.Connection) -> Dict[str, str]:
    """
    Maps post id -> combined post title/body.
    Used only as fallback context.
    """

    if not table_exists(core_conn, "posts"):
        return {}

    cols = get_columns(core_conn, "posts")
    if "id" not in cols:
        return {}

    has_title = "title" in cols
    has_body = "body" in cols

    select_cols = ["id"]
    if has_title:
        select_cols.append("title")
    if has_body:
        select_cols.append("body")

    df = pd.read_sql_query(
        f"""
        SELECT {", ".join(quote_ident(c) for c in select_cols)}
        FROM posts
        """,
        core_conn,
    )

    out = {}

    for _, row in df.iterrows():
        parts = []

        if has_title and not is_empty_text(row.get("title")):
            parts.append(f"Post title: {clean_text(row['title'])}")

        if has_body and not is_empty_text(row.get("body")):
            parts.append(f"Post body: {clean_text(row['body'])}")

        context = " ".join(parts).strip()

        if context:
            out[to_str_id(row["id"])] = context

    return out


def choose_best_context(
    row: pd.Series,
    parent_context_map: Dict[str, str],
    post_context_map: Dict[str, str],
    max_context_chars: int,
) -> Tuple[str, str]:
    """
    Priority:
    1. Final_context_extracted
    2. weighted_context
    3. target_context
    4. parent text from parent_child_similarity
    5. post title/body
    """

    priority = [
        ("Final_context_extracted", "comments.Final_context_extracted"),
        ("weighted_context", "comments.weighted_context"),
        ("target_context", "comments.target_context"),
    ]

    for col, source in priority:
        if col in row.index and not is_empty_text(row[col]):
            return truncate_text(row[col], max_context_chars), source

    comment_id = row.get("comment_id_str", "")
    if comment_id in parent_context_map:
        return truncate_text(parent_context_map[comment_id], max_context_chars), "parent_child_similarity.parent_text"

    post_id = row.get("post_id_str", "")
    if post_id in post_context_map:
        return truncate_text(post_context_map[post_id], max_context_chars), "posts.title_body"

    return "", "no_context_available"


# ==================================================
# AUXILIARY DATA
# ==================================================

def safe_read_table(conn: sqlite3.Connection, table_name: str) -> Optional[pd.DataFrame]:
    if not table_exists(conn, table_name):
        return None
    return pd.read_sql_query(f"SELECT * FROM {quote_ident(table_name)}", conn)


def build_auxiliary_maps(aux_conn: sqlite3.Connection, topic_conn: Optional[sqlite3.Connection]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    maps: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ------------------------------
    # Agent predictions
    # ------------------------------
    agent_map: Dict[str, Dict[str, Any]] = {}

    agent_df = safe_read_table(aux_conn, "agent_predictions")
    if agent_df is not None and "id" in agent_df.columns:
        for _, row in agent_df.iterrows():
            cid = to_str_id(row["id"])
            agent_map[cid] = {
                "predicted_agent": row.get("predicted_agent"),
                "agent_prediction_confidence": row.get("prediction_confidence"),
                "prob_chatgpt": row.get("prob_chatgpt"),
                "prob_claude": row.get("prob_claude"),
                "prob_deepseek": row.get("prob_deepseek"),
                "prob_llama": row.get("prob_llama"),
            }

    maps["agent"] = agent_map

    # ------------------------------
    # Cluster / sycophancy
    # ------------------------------
    cluster_map: Dict[str, Dict[str, Any]] = {}

    cluster_source = None
    if table_exists(aux_conn, "comment_cluster_agent_mapping"):
        cluster_source = "comment_cluster_agent_mapping"
    elif table_exists(aux_conn, "comment_clusters"):
        cluster_source = "comment_clusters"

    if cluster_source:
        cluster_df = safe_read_table(aux_conn, cluster_source)

        if cluster_df is not None and "comment_id" in cluster_df.columns:
            for _, row in cluster_df.iterrows():
                cid = to_str_id(row["comment_id"])
                cluster_map[cid] = {
                    "cluster": row.get("cluster"),
                    "cluster_name": row.get("cluster_name"),
                    "sycophancy_keyword_count": row.get("sycophancy_keyword_count"),
                    "sycophancy_matched_keywords": row.get("sycophancy_matched_keywords"),
                    "has_sycophancy_keyword": row.get("has_sycophancy_keyword"),
                    "is_most_sycophantic_cluster": row.get("is_most_sycophantic_cluster"),
                }

    maps["cluster"] = cluster_map

    # ------------------------------
    # Author cluster presence
    # ------------------------------
    author_map: Dict[str, Dict[str, Any]] = {}

    author_df = safe_read_table(aux_conn, "author_cluster_presence")
    if author_df is not None and "author" in author_df.columns:
        for _, row in author_df.iterrows():
            author = clean_text(row["author"])
            author_map[author] = {
                "author_total_comments_in_dataset": row.get("total_comments_in_dataset"),
                "author_clustered_comments": row.get("clustered_comments"),
                "author_num_clusters_with_comments": row.get("num_clusters_with_comments"),
                "author_clusters_with_comments": row.get("clusters_with_comments"),
            }

    maps["author"] = author_map

    # ------------------------------
    # Topic model
    # ------------------------------
    topic_map: Dict[str, Dict[str, Any]] = {}

    if topic_conn is not None and table_exists(topic_conn, "nmf_all_comment_topics"):
        topic_df = safe_read_table(topic_conn, "nmf_all_comment_topics")

        if topic_df is not None and "comment_id" in topic_df.columns:
            for _, row in topic_df.iterrows():
                cid = to_str_id(row["comment_id"])
                topic_map[cid] = {
                    "dominant_topic": row.get("dominant_topic"),
                    "topic_weight": row.get("topic_weight"),
                    "topic_confidence": row.get("topic_confidence"),
                }

    maps["topic"] = topic_map

    return maps


def enrich_comment_row(row: pd.Series, aux_maps: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    cid = row.get("comment_id_str", "")
    author = clean_text(row.get("author", ""))

    out: Dict[str, Any] = {}

    for map_name in ["agent", "cluster", "topic"]:
        out.update(aux_maps.get(map_name, {}).get(cid, {}))

    out.update(aux_maps.get("author", {}).get(author, {}))

    return out


# ==================================================
# NLI
# ==================================================

def get_label_mapping(model) -> Dict[str, int]:
    id2label = model.config.id2label
    label_to_index: Dict[str, int] = {}

    for idx, label in id2label.items():
        idx = int(idx)
        label_lower = str(label).lower()

        if "contradiction" in label_lower:
            label_to_index["contradiction"] = idx
        elif "neutral" in label_lower:
            label_to_index["neutral"] = idx
        elif "entailment" in label_lower:
            label_to_index["entailment"] = idx

    if set(label_to_index.keys()) == {"contradiction", "neutral", "entailment"}:
        return label_to_index

    return {
        "contradiction": 0,
        "neutral": 1,
        "entailment": 2,
    }


def load_nli_model(model_name: str, device: torch.device, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )

    model.to(device)
    model.eval()

    label_to_index = get_label_mapping(model)

    log(f"NLI model: {model_name}")
    log(f"Device: {device}")
    log(f"Label mapping: {label_to_index}")

    return tokenizer, model, label_to_index


def predict_nli_pairs(
    premises: List[str],
    hypotheses: List[str],
    tokenizer,
    model,
    device: torch.device,
    label_to_index: Dict[str, int],
    batch_size: int,
    max_length: int,
    desc: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(premises), batch_size), desc=desc):
        batch_premises = premises[start:start + batch_size]
        batch_hypotheses = hypotheses[start:start + batch_size]

        encoded = tokenizer(
            batch_premises,
            batch_hypotheses,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for row in probs:
            contradiction = float(row[label_to_index["contradiction"]])
            neutral = float(row[label_to_index["neutral"]])
            entailment = float(row[label_to_index["entailment"]])

            scores = {
                "contradiction": contradiction,
                "neutral": neutral,
                "entailment": entailment,
            }

            prediction = max(scores, key=scores.get)

            results.append(
                {
                    "contradiction_score": contradiction,
                    "neutral_score": neutral,
                    "entailment_score": entailment,
                    "prediction": prediction,
                    "max_nli_confidence": max(scores.values()),
                }
            )

    return results


# ==================================================
# RESULT BUILDING
# ==================================================

def build_base_result_rows(
    comments_df: pd.DataFrame,
    parent_context_map: Dict[str, str],
    post_context_map: Dict[str, str],
    aux_maps: Dict[str, Dict[str, Dict[str, Any]]],
    max_context_chars: int,
    max_comment_chars: int,
) -> pd.DataFrame:
    rows = []

    for _, row in comments_df.iterrows():
        comment_text = truncate_text(row.get("body", ""), max_comment_chars)

        if is_empty_text(comment_text):
            continue

        context_text, context_source = choose_best_context(
            row=row,
            parent_context_map=parent_context_map,
            post_context_map=post_context_map,
            max_context_chars=max_context_chars,
        )

        if is_empty_text(context_text):
            continue

        out = {
            "comment_id": row.get("id"),
            "comment_id_str": row.get("comment_id_str"),
            "post_id": row.get("post_id", None),
            "parent_comment_id": row.get("parent_comment_id", None),
            "depth": row.get("depth", None),
            "comment_order": row.get("comment_order", None),
            "author": row.get("author", None),
            "comment_text": comment_text,
            "context_text": context_text,
            "context_source": context_source,
            "context_type": row.get("context_type", None),
            "context_confidence": row.get("context_confidence", None),
            "context_confidence_bucket": confidence_bucket(row.get("context_confidence", None)),
            "context_ambiguities": row.get("context_ambiguities", None),
            "resolved_references": row.get("resolved_references", None),
            "extraction_method": row.get("extraction_method", None),
            "LLM_Stance_detection": row.get("LLM_Stance_detection", None),
            "LLM_Stance_detection_explanation": row.get("LLM_Stance_detection_explanation", None),
        }

        out.update(enrich_comment_row(row, aux_maps))

        rows.append(out)

    return pd.DataFrame(rows)


def context_alignment_label(prediction: str) -> str:
    if prediction == "entailment":
        return "context_aligned"
    if prediction == "contradiction":
        return "context_contradicting"
    if prediction == "neutral":
        return "context_neutral_or_new_information"
    return "unknown"


def add_nli_results(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    label_to_index: Dict[str, int],
    batch_size: int,
    max_length: int,
    bidirectional: bool,
) -> pd.DataFrame:
    if len(df) == 0:
        return df

    contexts = df["context_text"].astype(str).tolist()
    comments = df["comment_text"].astype(str).tolist()

    forward = predict_nli_pairs(
        premises=contexts,
        hypotheses=comments,
        tokenizer=tokenizer,
        model=model,
        device=device,
        label_to_index=label_to_index,
        batch_size=batch_size,
        max_length=max_length,
        desc="NLI context -> comment",
    )

    df["ctx_to_comment_contradiction_score"] = [x["contradiction_score"] for x in forward]
    df["ctx_to_comment_neutral_score"] = [x["neutral_score"] for x in forward]
    df["ctx_to_comment_entailment_score"] = [x["entailment_score"] for x in forward]
    df["ctx_to_comment_prediction"] = [x["prediction"] for x in forward]
    df["ctx_to_comment_max_confidence"] = [x["max_nli_confidence"] for x in forward]

    df["context_grounding_score"] = (
        df["ctx_to_comment_entailment_score"] -
        df["ctx_to_comment_contradiction_score"]
    )

    df["context_alignment_label"] = df["ctx_to_comment_prediction"].map(context_alignment_label)

    if bidirectional:
        backward = predict_nli_pairs(
            premises=comments,
            hypotheses=contexts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            label_to_index=label_to_index,
            batch_size=batch_size,
            max_length=max_length,
            desc="NLI comment -> context",
        )

        df["comment_to_ctx_contradiction_score"] = [x["contradiction_score"] for x in backward]
        df["comment_to_ctx_neutral_score"] = [x["neutral_score"] for x in backward]
        df["comment_to_ctx_entailment_score"] = [x["entailment_score"] for x in backward]
        df["comment_to_ctx_prediction"] = [x["prediction"] for x in backward]
        df["comment_to_ctx_max_confidence"] = [x["max_nli_confidence"] for x in backward]

        df["semantic_equivalence_signal"] = (
            (df["ctx_to_comment_prediction"] == "entailment") &
            (df["comment_to_ctx_prediction"] == "entailment")
        ).astype(int)

    return df


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
            mean_grounding_score=("context_grounding_score", "mean"),
            median_grounding_score=("context_grounding_score", "median"),
            mean_entailment=("ctx_to_comment_entailment_score", "mean"),
            mean_contradiction=("ctx_to_comment_contradiction_score", "mean"),
            mean_neutral=("ctx_to_comment_neutral_score", "mean"),
            entailment_rate=("ctx_to_comment_prediction", lambda x: (x == "entailment").mean()),
            contradiction_rate=("ctx_to_comment_prediction", lambda x: (x == "contradiction").mean()),
            neutral_rate=("ctx_to_comment_prediction", lambda x: (x == "neutral").mean()),
            avg_context_confidence=("context_confidence", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        )
        .reset_index()
        .sort_values(["n_comments", "mean_grounding_score"], ascending=[False, False])
    )

    return agg


def save_output(output_db: Path, results_df: pd.DataFrame) -> None:
    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(str(output_db))

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.commit()

        results_df.to_sql(
            "comment_context_nli_results",
            conn,
            if_exists="replace",
            index=False,
        )

        aggregations = {
            "agg_by_context_source": "context_source",
            "agg_by_context_type": "context_type",
            "agg_by_context_confidence_bucket": "context_confidence_bucket",
            "agg_by_author": "author",
            "agg_by_predicted_agent": "predicted_agent",
            "agg_by_cluster": "cluster",
            "agg_by_stance_detection": "LLM_Stance_detection",
            "agg_by_topic": "dominant_topic",
            "agg_by_alignment_label": "context_alignment_label",
        }

        for table_name, group_col in aggregations.items():
            agg_df = aggregate_by(results_df, group_col)
            if len(agg_df) > 0:
                agg_df.to_sql(table_name, conn, if_exists="replace", index=False)

        index_sql = [
            "CREATE INDEX IF NOT EXISTS idx_results_comment_id ON comment_context_nli_results (comment_id_str)",
            "CREATE INDEX IF NOT EXISTS idx_results_author ON comment_context_nli_results (author)",
            "CREATE INDEX IF NOT EXISTS idx_results_agent ON comment_context_nli_results (predicted_agent)",
            "CREATE INDEX IF NOT EXISTS idx_results_cluster ON comment_context_nli_results (cluster)",
            "CREATE INDEX IF NOT EXISTS idx_results_topic ON comment_context_nli_results (dominant_topic)",
            "CREATE INDEX IF NOT EXISTS idx_results_alignment ON comment_context_nli_results (context_alignment_label)",
            "CREATE INDEX IF NOT EXISTS idx_results_context_type ON comment_context_nli_results (context_type)",
        ]

        for sql in index_sql:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass

        conn.commit()

        log("\nOutput tables:")
        for table in get_tables(conn):
            count = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()[0]
            log(f"- {table}: {count:,} rows")

    finally:
        conn.close()


# ==================================================
# MAIN
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rich NLI between extracted context and comment body."
    )

    parser.add_argument("--core-db", default="moltbook_final_NLI_sim_1500_per_table.db")
    parser.add_argument("--aux-db", default="moltbook_merged_4_dbs_all_tables.db")
    parser.add_argument("--topic-db", default="topic_model_nmf_all_comments_hpc.db")
    parser.add_argument("--output-db", default="comment_context_nli_rich_analysis.db")

    parser.add_argument("--nli-model", default="typeform/distilbert-base-uncased-mnli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-context-chars", type=int, default=1600)
    parser.add_argument("--max-comment-chars", type=int, default=900)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-bidirectional", action="store_true")

    args = parser.parse_args()

    core_db = Path(args.core_db).resolve()
    aux_db = Path(args.aux_db).resolve()
    topic_db = Path(args.topic_db).resolve()
    output_db = Path(args.output_db).resolve()

    if not core_db.exists():
        raise FileNotFoundError(f"Core DB not found: {core_db}")

    if not aux_db.exists():
        raise FileNotFoundError(f"Aux DB not found: {aux_db}")

    if not topic_db.exists():
        log(f"Warning: topic DB not found: {topic_db}")
        log("Continuing without topic enrichment.")
        topic_db = None

    log(f"Core DB: {core_db}")
    log(f"Aux DB: {aux_db}")
    log(f"Topic DB: {topic_db if topic_db else 'not used'}")
    log(f"Output DB: {output_db}")

    core_conn = sqlite3.connect(str(core_db))
    aux_conn = sqlite3.connect(str(aux_db))
    topic_conn = sqlite3.connect(str(topic_db)) if topic_db is not None else None

    try:
        log("\nLoading comments and context fields from core DB...")
        comments_df = load_comments(core_conn, args.limit)
        log(f"Comments loaded: {len(comments_df):,}")

        log("Loading fallback parent contexts...")
        parent_context_map = load_parent_context_map(core_conn)
        log(f"Parent contexts loaded: {len(parent_context_map):,}")

        log("Loading fallback post contexts...")
        post_context_map = load_post_context_map(core_conn)
        log(f"Post contexts loaded: {len(post_context_map):,}")

        log("Loading auxiliary maps...")
        aux_maps = build_auxiliary_maps(aux_conn, topic_conn)

        for name, m in aux_maps.items():
            log(f"- {name}: {len(m):,}")

        log("\nBuilding comment-level context rows...")
        results_df = build_base_result_rows(
            comments_df=comments_df,
            parent_context_map=parent_context_map,
            post_context_map=post_context_map,
            aux_maps=aux_maps,
            max_context_chars=args.max_context_chars,
            max_comment_chars=args.max_comment_chars,
        )

        log(f"Rows with usable comment and context: {len(results_df):,}")

        if len(results_df) == 0:
            raise RuntimeError("No usable comment/context pairs found.")

        device = choose_device(force_cpu=args.cpu)

        tokenizer, model, label_to_index = load_nli_model(
            model_name=args.nli_model,
            device=device,
            local_files_only=args.local_files_only,
        )

        results_df = add_nli_results(
            df=results_df,
            tokenizer=tokenizer,
            model=model,
            device=device,
            label_to_index=label_to_index,
            batch_size=args.batch_size,
            max_length=args.max_length,
            bidirectional=not args.no_bidirectional,
        )

        log("\nSaving final output...")
        save_output(output_db, results_df)

        log("\nDone.")
        log(f"Final output DB: {output_db}")

    finally:
        core_conn.close()
        aux_conn.close()
        if topic_conn is not None:
            topic_conn.close()
        log("Input database connections closed.")


if __name__ == "__main__":
    main()
