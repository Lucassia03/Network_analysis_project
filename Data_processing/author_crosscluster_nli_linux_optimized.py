import argparse
import hashlib
import itertools
import os
import re
import sqlite3
import string
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Linux/HuggingFace safety: prevents tokenizer thread oversubscription warnings/deadlocks.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ==================================================
# DEFAULT CONFIG
# ==================================================

DEFAULT_DB_PATH = "moltbook_final_NLI_sim_WITH_CLUSTER_AGENT_MAPPING.db"
DEFAULT_OUTPUT_DB = "author_crosscluster_nli_linux_optimized.db"
DEFAULT_OUTPUT_TABLE = "author_crosscluster_nli_fast_summary"

DEFAULT_N_CLUSTERS = 7
DEFAULT_RANDOM_STATE = 42
DEFAULT_NLI_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
DEFAULT_NLI_BATCH_SIZE = 16
DEFAULT_MAX_TOKEN_LENGTH = 256
DEFAULT_MAX_PAIRS_PER_DIRECTION = 500

CUSTOM_STOPWORDS = {
    "im", "ive", "id", "youre", "youve", "theyre", "thats", "dont",
    "didnt", "doesnt", "isnt", "arent", "wasnt", "werent", "cant",
    "couldnt", "wouldnt", "shouldnt", "ill", "theres", "heres",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)


# ==================================================
# UTILS
# ==================================================

def log(message: str) -> None:
    print(message, flush=True)


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def stable_seed(text: str) -> int:
    value = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(value[:8], 16)


def clean_text_for_clustering(text: object) -> str:
    """Aggressive normalization for TF-IDF clustering."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [tok for tok in text.split() if tok not in STOP_WORDS and len(tok) > 2]
    return " ".join(tokens)


def normalize_for_nli(text: object) -> str:
    """Light normalization for the NLI model. Keep original wording."""
    return re.sub(r"\s+", " ", str(text)).strip()


def choose_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Mostly relevant for macOS, harmless on Linux.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==================================================
# SQLITE
# ==================================================

def connect_sqlite(path: str, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        uri = f"file:{Path(path).resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=60)
    else:
        conn = sqlite3.connect(path, timeout=60)

    conn.row_factory = sqlite3.Row
    return conn


def configure_output_sqlite(conn: sqlite3.Connection) -> None:
    # WAL is more robust for incremental writes on Linux.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.commit()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def load_comments(db_path: str, min_body_chars: int = 1) -> pd.DataFrame:
    """Load only columns needed by this pipeline."""
    conn = connect_sqlite(db_path, readonly=True)
    try:
        if not table_exists(conn, "comments"):
            raise RuntimeError("Input DB does not contain a 'comments' table.")

        df = pd.read_sql_query(
            """
            SELECT
                id AS comment_id,
                post_id,
                author,
                body
            FROM comments
            WHERE body IS NOT NULL
              AND TRIM(body) != ''
              AND LENGTH(TRIM(body)) >= ?
            """,
            conn,
            params=(int(min_body_chars),),
        )
    finally:
        conn.close()

    return df


def create_output_table(conn: sqlite3.Connection, table_name: str, replace: bool = False) -> None:
    if replace:
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table_name)}")

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {quote_ident(table_name)} (
            author TEXT NOT NULL,
            total_clustered_comments INTEGER NOT NULL,
            num_clusters_with_comments INTEGER NOT NULL,
            clusters_with_comments TEXT NOT NULL,

            premise_cluster INTEGER NOT NULL,
            hypothesis_cluster INTEGER NOT NULL,
            direction TEXT NOT NULL,

            comments_in_premise_cluster INTEGER NOT NULL,
            comments_in_hypothesis_cluster INTEGER NOT NULL,

            total_possible_pairs INTEGER NOT NULL,
            processed_pairs INTEGER NOT NULL,
            sampling_used INTEGER NOT NULL,

            contradiction_count INTEGER NOT NULL,
            neutral_count INTEGER NOT NULL,
            entailment_count INTEGER NOT NULL,

            contradiction_rate REAL NOT NULL,
            neutral_rate REAL NOT NULL,
            entailment_rate REAL NOT NULL,

            avg_contradiction_score REAL NOT NULL,
            avg_neutral_score REAL NOT NULL,
            avg_entailment_score REAL NOT NULL,

            dominant_nli_tag TEXT NOT NULL,

            PRIMARY KEY (author, premise_cluster, hypothesis_cluster)
        )
        """
    )
    conn.commit()


def output_row_exists(
    conn: sqlite3.Connection,
    table_name: str,
    author: str,
    premise_cluster: int,
    hypothesis_cluster: int,
) -> bool:
    row = conn.execute(
        f"""
        SELECT 1
        FROM {quote_ident(table_name)}
        WHERE author = ?
          AND premise_cluster = ?
          AND hypothesis_cluster = ?
        LIMIT 1
        """,
        (author, int(premise_cluster), int(hypothesis_cluster)),
    ).fetchone()
    return row is not None


def insert_summary_rows(
    conn: sqlite3.Connection,
    table_name: str,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
        return

    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_sql = ", ".join(quote_ident(col) for col in columns)

    sql = f"""
        INSERT OR REPLACE INTO {quote_ident(table_name)} ({col_sql})
        VALUES ({placeholders})
    """

    values = [tuple(row[col] for col in columns) for row in rows]
    conn.executemany(sql, values)
    conn.commit()


# ==================================================
# CLUSTERING
# ==================================================

def add_clusters(
    df: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    use_minibatch: bool,
    min_words_after_cleaning: int,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_max_df: float,
) -> pd.DataFrame:
    df = df.copy()

    df["author"] = df["author"].fillna("unknown_author").astype(str).str.strip()
    df.loc[df["author"] == "", "author"] = "unknown_author"

    df["cleaned_text"] = df["body"].astype(str).map(clean_text_for_clustering)

    df = df[df["cleaned_text"].notna() & (df["cleaned_text"].str.strip() != "")].copy()
    df = df[df["cleaned_text"].str.split().str.len() >= int(min_words_after_cleaning)].copy()
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No comments left after cleaning/filtering.")

    effective_clusters = min(int(n_clusters), len(df))
    if effective_clusters < 2:
        raise RuntimeError("Need at least 2 valid comments/clusters for cross-cluster analysis.")

    vectorizer = TfidfVectorizer(
        max_features=int(tfidf_max_features),
        min_df=int(tfidf_min_df),
        max_df=float(tfidf_max_df),
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=np.float32,
    )

    log("Vectorizing comments with TF-IDF...")
    X = vectorizer.fit_transform(df["cleaned_text"])

    log(f"Clustering with {'MiniBatchKMeans' if use_minibatch else 'KMeans'}...")
    if use_minibatch:
        clusterer = MiniBatchKMeans(
            n_clusters=effective_clusters,
            random_state=int(random_state),
            n_init=10,
            batch_size=2048,
            reassignment_ratio=0.01,
        )
    else:
        clusterer = KMeans(
            n_clusters=effective_clusters,
            random_state=int(random_state),
            n_init=20,
            algorithm="lloyd",
        )

    df["cluster"] = clusterer.fit_predict(X).astype(int)
    df["nli_text"] = df["body"].astype(str).map(normalize_for_nli)

    return df


def get_multicluster_author_info(df: pd.DataFrame) -> pd.DataFrame:
    author_info = (
        df.groupby("author")
        .agg(
            total_clustered_comments=("comment_id", "count"),
            num_clusters_with_comments=("cluster", "nunique"),
            clusters_with_comments=("cluster", lambda x: ", ".join(map(str, sorted(x.unique())))),
        )
        .reset_index()
    )

    return author_info[author_info["num_clusters_with_comments"] > 1].copy()


# ==================================================
# NLI MODEL
# ==================================================

def get_label_mapping(model: AutoModelForSequenceClassification) -> Dict[str, int]:
    id2label = model.config.id2label
    label_to_index: Dict[str, int] = {}

    for idx, label in id2label.items():
        label_lower = str(label).lower()
        idx_int = int(idx)

        if "contradiction" in label_lower:
            label_to_index["contradiction"] = idx_int
        elif "neutral" in label_lower:
            label_to_index["neutral"] = idx_int
        elif "entailment" in label_lower:
            label_to_index["entailment"] = idx_int

    required = {"contradiction", "neutral", "entailment"}
    if set(label_to_index.keys()) == required:
        return label_to_index

    # Common MNLI fallback.
    return {"contradiction": 0, "neutral": 1, "entailment": 2}


def load_nli_model(
    model_name: str,
    device: torch.device,
    local_files_only: bool,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, Dict[str, int]]:
    log(f"Loading NLI model: {model_name}")
    log(f"Device: {device}")

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
    log(f"Label mapping: {label_to_index}")

    return tokenizer, model, label_to_index


def predict_nli_batch(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    label_to_index: Dict[str, int],
    max_token_length: int,
) -> List[Dict[str, object]]:
    encoded = tokenizer(
        list(premises),
        list(hypotheses),
        padding=True,
        truncation=True,
        max_length=int(max_token_length),
        return_tensors="pt",
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

    results: List[Dict[str, object]] = []

    for row in probs:
        contradiction_score = float(row[label_to_index["contradiction"]])
        neutral_score = float(row[label_to_index["neutral"]])
        entailment_score = float(row[label_to_index["entailment"]])

        scores = {
            "contradiction": contradiction_score,
            "neutral": neutral_score,
            "entailment": entailment_score,
        }
        prediction = max(scores, key=scores.get)

        results.append(
            {
                "contradiction_score": contradiction_score,
                "neutral_score": neutral_score,
                "entailment_score": entailment_score,
                "prediction": prediction,
            }
        )

    return results


# ==================================================
# PAIR SAMPLING / PROCESSING
# ==================================================

def sample_pair_indices(
    n_a: int,
    n_b: int,
    seed_key: str,
    max_pairs_per_direction: Optional[int],
) -> Optional[List[Tuple[int, int]]]:
    total_pairs = int(n_a) * int(n_b)

    if total_pairs == 0:
        return []

    if max_pairs_per_direction is None:
        return None

    if total_pairs <= int(max_pairs_per_direction):
        return None

    rng = np.random.default_rng(stable_seed(seed_key))
    sampled_flat_indices = rng.choice(total_pairs, size=int(max_pairs_per_direction), replace=False)

    return [(int(flat_idx // n_b), int(flat_idx % n_b)) for flat_idx in sampled_flat_indices]


def process_direction(
    author: str,
    premise_cluster: int,
    hypothesis_cluster: int,
    premise_texts: Sequence[str],
    hypothesis_texts: Sequence[str],
    pair_indices: Optional[Sequence[Tuple[int, int]]],
    num_clusters_with_comments: int,
    clusters_with_comments: str,
    total_clustered_comments: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    label_to_index: Dict[str, int],
    nli_batch_size: int,
    max_token_length: int,
) -> Optional[Dict[str, object]]:
    total_possible_pairs = len(premise_texts) * len(hypothesis_texts)
    if total_possible_pairs == 0:
        return None

    if pair_indices is None:
        pairs_iter: Iterable[Tuple[int, int]] = itertools.product(
            range(len(premise_texts)),
            range(len(hypothesis_texts)),
        )
        expected_pairs = total_possible_pairs
    else:
        pairs_iter = iter(pair_indices)
        expected_pairs = len(pair_indices)

    counts = {"contradiction": 0, "neutral": 0, "entailment": 0}
    score_sums = {"contradiction": 0.0, "neutral": 0.0, "entailment": 0.0}

    batch_premises: List[str] = []
    batch_hypotheses: List[str] = []
    processed_pairs = 0

    def flush_batch() -> None:
        nonlocal processed_pairs
        if not batch_premises:
            return

        predictions = predict_nli_batch(
            premises=batch_premises,
            hypotheses=batch_hypotheses,
            tokenizer=tokenizer,
            model=model,
            device=device,
            label_to_index=label_to_index,
            max_token_length=max_token_length,
        )

        for pred in predictions:
            label = str(pred["prediction"])
            counts[label] += 1
            score_sums["contradiction"] += float(pred["contradiction_score"])
            score_sums["neutral"] += float(pred["neutral_score"])
            score_sums["entailment"] += float(pred["entailment_score"])
            processed_pairs += 1

        batch_premises.clear()
        batch_hypotheses.clear()

    for i, j in pairs_iter:
        batch_premises.append(premise_texts[i])
        batch_hypotheses.append(hypothesis_texts[j])

        if len(batch_premises) >= int(nli_batch_size):
            flush_batch()

    flush_batch()

    if processed_pairs == 0:
        contradiction_rate = neutral_rate = entailment_rate = 0.0
        avg_contradiction_score = avg_neutral_score = avg_entailment_score = 0.0
        dominant_nli_tag = "no_pairs"
    else:
        contradiction_rate = counts["contradiction"] / processed_pairs
        neutral_rate = counts["neutral"] / processed_pairs
        entailment_rate = counts["entailment"] / processed_pairs

        avg_contradiction_score = score_sums["contradiction"] / processed_pairs
        avg_neutral_score = score_sums["neutral"] / processed_pairs
        avg_entailment_score = score_sums["entailment"] / processed_pairs

        dominant_nli_tag = max(counts, key=counts.get)

    return {
        "author": author,
        "total_clustered_comments": int(total_clustered_comments),
        "num_clusters_with_comments": int(num_clusters_with_comments),
        "clusters_with_comments": clusters_with_comments,

        "premise_cluster": int(premise_cluster),
        "hypothesis_cluster": int(hypothesis_cluster),
        "direction": f"{premise_cluster} -> {hypothesis_cluster}",

        "comments_in_premise_cluster": int(len(premise_texts)),
        "comments_in_hypothesis_cluster": int(len(hypothesis_texts)),

        "total_possible_pairs": int(total_possible_pairs),
        "processed_pairs": int(processed_pairs),
        "sampling_used": int(processed_pairs < total_possible_pairs),

        "contradiction_count": int(counts["contradiction"]),
        "neutral_count": int(counts["neutral"]),
        "entailment_count": int(counts["entailment"]),

        "contradiction_rate": float(contradiction_rate),
        "neutral_rate": float(neutral_rate),
        "entailment_rate": float(entailment_rate),

        "avg_contradiction_score": float(avg_contradiction_score),
        "avg_neutral_score": float(avg_neutral_score),
        "avg_entailment_score": float(avg_entailment_score),

        "dominant_nli_tag": dominant_nli_tag,
    }


# ==================================================
# MAIN PIPELINE
# ==================================================

def run(args: argparse.Namespace) -> None:
    if not Path(args.db_path).exists():
        raise FileNotFoundError(f"Input database not found: {args.db_path}")

    log("Loading comments...")
    comments = load_comments(args.db_path, min_body_chars=args.min_body_chars)
    log(f"Loaded comments: {len(comments):,}")

    log("Generating clusters...")
    comments = add_clusters(
        df=comments,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        use_minibatch=args.minibatch_kmeans,
        min_words_after_cleaning=args.min_words_after_cleaning,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_max_df=args.tfidf_max_df,
    )
    log(f"Clustered comments: {len(comments):,}")

    log("Selecting multi-cluster authors...")
    author_info = get_multicluster_author_info(comments)
    log(f"Multi-cluster authors: {len(author_info):,}")

    if len(author_info) == 0:
        log("No multi-cluster authors found. Nothing to process.")
        return

    comments = comments.merge(author_info, on="author", how="inner")

    device = choose_device(force_cpu=args.cpu)
    tokenizer, model, label_to_index = load_nli_model(
        model_name=args.nli_model_name,
        device=device,
        local_files_only=args.local_files_only,
    )

    conn_out = connect_sqlite(args.output_db, readonly=False)
    try:
        configure_output_sqlite(conn_out)
        create_output_table(conn_out, args.output_table, replace=args.replace_output)

        authors = author_info["author"].astype(str).tolist()

        for author_index, author in enumerate(authors, start=1):
            author_df = comments[comments["author"] == author].copy()
            clusters = sorted(int(x) for x in author_df["cluster"].unique())

            num_clusters_with_comments = int(author_df["num_clusters_with_comments"].iloc[0])
            clusters_with_comments = str(author_df["clusters_with_comments"].iloc[0])
            total_clustered_comments = int(author_df["total_clustered_comments"].iloc[0])

            log(
                f"Author {author_index:,}/{len(authors):,} | "
                f"{author} | clusters: {clusters_with_comments} | "
                f"comments: {total_clustered_comments:,}"
            )

            cluster_to_texts: Dict[int, List[str]] = {}
            for cluster_id in clusters:
                texts = (
                    author_df.loc[author_df["cluster"] == cluster_id, "nli_text"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
                texts = [txt for txt in texts if txt.strip()]
                cluster_to_texts[cluster_id] = texts

            pending_rows: List[Dict[str, object]] = []

            for cluster_a, cluster_b in itertools.combinations(clusters, 2):
                texts_a = cluster_to_texts[cluster_a]
                texts_b = cluster_to_texts[cluster_b]

                if not texts_a or not texts_b:
                    continue

                seed_key = f"{author}|{cluster_a}|{cluster_b}|{args.random_state}"
                pair_indices_ab = sample_pair_indices(
                    len(texts_a),
                    len(texts_b),
                    seed_key,
                    args.max_pairs_per_direction,
                )

                if pair_indices_ab is None:
                    pair_indices_ba = None
                else:
                    pair_indices_ba = [(j, i) for i, j in pair_indices_ab]

                # Direction A -> B
                if not args.resume or not output_row_exists(conn_out, args.output_table, author, cluster_a, cluster_b):
                    row_ab = process_direction(
                        author=author,
                        premise_cluster=cluster_a,
                        hypothesis_cluster=cluster_b,
                        premise_texts=texts_a,
                        hypothesis_texts=texts_b,
                        pair_indices=pair_indices_ab,
                        num_clusters_with_comments=num_clusters_with_comments,
                        clusters_with_comments=clusters_with_comments,
                        total_clustered_comments=total_clustered_comments,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        label_to_index=label_to_index,
                        nli_batch_size=args.nli_batch_size,
                        max_token_length=args.max_token_length,
                    )
                    if row_ab is not None:
                        pending_rows.append(row_ab)

                # Direction B -> A
                if not args.resume or not output_row_exists(conn_out, args.output_table, author, cluster_b, cluster_a):
                    row_ba = process_direction(
                        author=author,
                        premise_cluster=cluster_b,
                        hypothesis_cluster=cluster_a,
                        premise_texts=texts_b,
                        hypothesis_texts=texts_a,
                        pair_indices=pair_indices_ba,
                        num_clusters_with_comments=num_clusters_with_comments,
                        clusters_with_comments=clusters_with_comments,
                        total_clustered_comments=total_clustered_comments,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        label_to_index=label_to_index,
                        nli_batch_size=args.nli_batch_size,
                        max_token_length=args.max_token_length,
                    )
                    if row_ba is not None:
                        pending_rows.append(row_ba)

                if len(pending_rows) >= args.commit_every_rows:
                    insert_summary_rows(conn_out, args.output_table, pending_rows)
                    log(f"  saved rows: {len(pending_rows)}")
                    pending_rows.clear()

            if pending_rows:
                insert_summary_rows(conn_out, args.output_table, pending_rows)
                log(f"  saved rows: {len(pending_rows)}")

        log("Finished.")
        log(f"Output DB: {args.output_db}")
        log(f"Output table: {args.output_table}")

    finally:
        conn_out.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linux-optimized author cross-cluster NLI analysis."
    )

    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--output-db", default=DEFAULT_OUTPUT_DB)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)

    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--minibatch-kmeans", action="store_true", help="Use MiniBatchKMeans instead of full KMeans.")

    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--tfidf-max-df", type=float, default=0.85)
    parser.add_argument("--min-words-after-cleaning", type=int, default=4)
    parser.add_argument("--min-body-chars", type=int, default=1)

    parser.add_argument("--nli-model-name", default=DEFAULT_NLI_MODEL_NAME)
    parser.add_argument("--nli-batch-size", type=int, default=DEFAULT_NLI_BATCH_SIZE)
    parser.add_argument("--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH)
    parser.add_argument("--max-pairs-per-direction", type=int, default=DEFAULT_MAX_PAIRS_PER_DIRECTION)
    parser.add_argument("--no-pair-limit", action="store_true", help="Disable pair sampling. Can be very slow.")

    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only locally cached HuggingFace files.")

    parser.add_argument("--replace-output", action="store_true", help="Drop and recreate output table.")
    parser.add_argument("--resume", action="store_true", help="Skip author/cluster directions already present in output table.")
    parser.add_argument("--commit-every-rows", type=int, default=20)

    args = parser.parse_args(argv)

    if args.no_pair_limit:
        args.max_pairs_per_direction = None

    if args.n_clusters < 2:
        parser.error("--n-clusters must be >= 2")

    if args.nli_batch_size < 1:
        parser.error("--nli-batch-size must be >= 1")

    if args.max_token_length < 8:
        parser.error("--max-token-length must be >= 8")

    if args.commit_every_rows < 1:
        parser.error("--commit-every-rows must be >= 1")

    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        sys.exit(1)
