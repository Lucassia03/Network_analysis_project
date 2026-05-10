import re
import json
import string
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
except ImportError as exc:
    raise ImportError("Installa nltk: pip install nltk") from exc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_DB = "moltbook_final_NLI_sim_WITH_CLUSTER_AGENT_MAPPING.db"
OUTPUT_DB = "topic_model_nmf_all_comments.db"

SOURCE_TABLE = "comments"

# Numero di topic latenti. Prova 10, 15, 20, 25.
N_TOPICS = 15

RANDOM_STATE = 42

# TF-IDF
MAX_FEATURES = 15000
MIN_DF = 5
MAX_DF = 0.80
NGRAM_RANGE = (1, 2)

# Filtri commenti
MIN_CLEAN_WORDS = 4

# Output interpretativo
TOP_TERMS_PER_TOPIC = 30
EXAMPLES_PER_TOPIC = 10

# Coordinate 2D per visualizzazione futura
SVD_COMPONENTS_FOR_PLOT = 2

# Metriche: calcolate su spazio SVD ridotto per evitare densificazione enorme
SVD_COMPONENTS_FOR_METRICS = 100
COMPUTE_METRICS = True


# --------------------------------------------------
# STOPWORDS
# --------------------------------------------------

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

STOP_WORDS = STOP_WORDS.union({
    "im", "ive", "id", "youre", "youve", "theyre", "thats", "dont",
    "didnt", "doesnt", "isnt", "arent", "wasnt", "werent", "cant",
    "couldnt", "wouldnt", "shouldnt", "ill", "theres", "heres",
    "amp", "nbsp", "lol", "yeah", "okay", "ok", "also", "really",
    "would", "could", "should", "get", "got", "one", "like", "just"
})


# --------------------------------------------------
# SQLITE UTILS
# --------------------------------------------------

def check_table_exists(db_path: str, table_name: str) -> None:
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database non trovato: {db_path.resolve()}")

    conn = sqlite3.connect(db_path)

    try:
        exists = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?;
            """,
            (table_name,),
        ).fetchone()

        if exists is None:
            raise ValueError(f"Tabella '{table_name}' non trovata in {db_path}")

    finally:
        conn.close()


def save_table(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    conn = sqlite3.connect(db_path)

    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    finally:
        conn.close()


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )

    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        tok for tok in text.split()
        if tok not in STOP_WORDS and len(tok) > 2
    ]

    return " ".join(tokens)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_comments(db_path: str) -> pd.DataFrame:
    check_table_exists(db_path, SOURCE_TABLE)

    conn = sqlite3.connect(db_path)

    try:
        query = """
        SELECT
            id AS comment_id,
            post_id,
            parent_comment_id,
            depth,
            comment_order,
            author,
            body,
            scraped_at
        FROM comments
        WHERE body IS NOT NULL
          AND TRIM(body) != '';
        """

        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    return df


# --------------------------------------------------
# TOPIC HELPERS
# --------------------------------------------------

def get_topic_terms(model: NMF, feature_names: np.ndarray, n_terms: int) -> dict:
    topic_terms = {}

    for topic_id, weights in enumerate(model.components_):
        top_idx = weights.argsort()[::-1][:n_terms]
        terms = []

        for rank, idx in enumerate(top_idx, start=1):
            terms.append({
                "topic": int(topic_id),
                "term_rank": int(rank),
                "term": str(feature_names[idx]),
                "weight": float(weights[idx]),
            })

        topic_terms[int(topic_id)] = terms

    return topic_terms


def build_topic_terms_table(topic_terms: dict) -> pd.DataFrame:
    rows = []

    for terms in topic_terms.values():
        rows.extend(terms)

    return pd.DataFrame(rows)


def build_topic_summary(
    df: pd.DataFrame,
    topic_terms: dict,
    examples_per_topic: int
) -> pd.DataFrame:
    rows = []
    n_total = len(df)

    for topic_id in sorted(df["dominant_topic"].unique()):
        group = df[df["dominant_topic"] == topic_id].copy()
        group = group.sort_values("topic_confidence", ascending=False)

        top_terms = [
            item["term"]
            for item in topic_terms.get(int(topic_id), [])
        ]

        examples = []

        for _, row in group.head(examples_per_topic).iterrows():
            examples.append({
                "comment_id": int(row["comment_id"]),
                "post_id": int(row["post_id"]),
                "depth": int(row["depth"]) if pd.notna(row["depth"]) else None,
                "topic_confidence": float(row["topic_confidence"]),
                "body": str(row["body"])[:900],
            })

        rows.append({
            "topic": int(topic_id),
            "n_comments_dominant": int(len(group)),
            "share_pct": float(len(group) / n_total * 100) if n_total else 0.0,
            "avg_topic_confidence": float(group["topic_confidence"].mean()),
            "median_topic_confidence": float(group["topic_confidence"].median()),
            "top_terms": json.dumps(top_terms, ensure_ascii=False),
            "example_comments": json.dumps(examples, ensure_ascii=False),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("n_comments_dominant", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:
    print("Loading all comments...")

    df = load_comments(INPUT_DB)

    print(f"Raw non-empty comments loaded: {len(df):,}")

    print("Cleaning text...")

    df["cleaned_text"] = df["body"].fillna("").astype(str).apply(clean_text)

    before = len(df)

    df = df[df["cleaned_text"].str.strip() != ""].copy()
    df = df[df["cleaned_text"].str.split().str.len() >= MIN_CLEAN_WORDS].copy()

    print(f"Comments after cleaning: {len(df):,}")
    print(f"Comments removed: {before - len(df):,}")

    if len(df) < N_TOPICS:
        raise ValueError(
            f"Commenti insufficienti ({len(df)}) per {N_TOPICS} topic."
        )

    print("Vectorizing with TF-IDF...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        norm="l2",
    )

    X = vectorizer.fit_transform(df["cleaned_text"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    print("TF-IDF matrix shape:", X.shape)

    print("Running NMF topic model...")

    nmf = NMF(
        n_components=N_TOPICS,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=500,
        random_state=RANDOM_STATE,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )

    W = nmf.fit_transform(X)

    dominant_topic = W.argmax(axis=1)
    dominant_weight = W.max(axis=1)
    row_sums = W.sum(axis=1)

    topic_confidence = np.divide(
        dominant_weight,
        row_sums,
        out=np.zeros_like(dominant_weight),
        where=row_sums != 0,
    )

    df["dominant_topic"] = dominant_topic.astype(int)
    df["topic_weight"] = dominant_weight.astype(float)
    df["topic_confidence"] = topic_confidence.astype(float)

    # Salva anche tutti i pesi topic come JSON per ogni commento.
    topic_weight_json = []

    for row in W:
        topic_weight_json.append(
            json.dumps(
                {f"topic_{i}": float(v) for i, v in enumerate(row)},
                ensure_ascii=False,
            )
        )

    df["topic_weights_json"] = topic_weight_json

    print("Topic sizes by dominant topic:")
    print(df["dominant_topic"].value_counts().sort_index().to_string())

    print("Extracting top terms...")

    topic_terms = get_topic_terms(
        model=nmf,
        feature_names=feature_names,
        n_terms=TOP_TERMS_PER_TOPIC,
    )

    for topic_id in sorted(topic_terms):
        terms = [item["term"] for item in topic_terms[topic_id][:15]]
        print(f"Topic {topic_id}: {', '.join(terms)}")

    print("Computing 2D SVD coordinates...")

    svd_2d = TruncatedSVD(
        n_components=SVD_COMPONENTS_FOR_PLOT,
        random_state=RANDOM_STATE,
    )

    coords = svd_2d.fit_transform(X)
    df["svd_x"] = coords[:, 0]
    df["svd_y"] = coords[:, 1]

    metrics = {}

    if COMPUTE_METRICS and df["dominant_topic"].nunique() > 1:
        print("Computing topic assignment quality metrics on SVD space...")

        n_metric_components = min(
            SVD_COMPONENTS_FOR_METRICS,
            X.shape[1] - 1,
            X.shape[0] - 1,
        )

        if n_metric_components >= 2:
            svd_metrics = TruncatedSVD(
                n_components=n_metric_components,
                random_state=RANDOM_STATE,
            )

            X_reduced = svd_metrics.fit_transform(X)
            labels = df["dominant_topic"].to_numpy()

            metrics = {
                "silhouette_cosine_on_svd": float(
                    silhouette_score(X_reduced, labels, metric="cosine")
                ),
                "davies_bouldin_on_svd": float(
                    davies_bouldin_score(X_reduced, labels)
                ),
                "calinski_harabasz_on_svd": float(
                    calinski_harabasz_score(X_reduced, labels)
                ),
                "svd_metric_components": int(n_metric_components),
            }

            print(json.dumps(metrics, indent=2))

    topic_terms_df = build_topic_terms_table(topic_terms)

    topic_summary_df = build_topic_summary(
        df=df,
        topic_terms=topic_terms,
        examples_per_topic=EXAMPLES_PER_TOPIC,
    )

    for key, value in metrics.items():
        topic_summary_df[key] = value

    comment_topics_df = df[
        [
            "comment_id",
            "post_id",
            "parent_comment_id",
            "depth",
            "comment_order",
            "author",
            "body",
            "cleaned_text",
            "dominant_topic",
            "topic_weight",
            "topic_confidence",
            "topic_weights_json",
            "svd_x",
            "svd_y",
            "scraped_at",
        ]
    ].copy()

    config_df = pd.DataFrame([
        {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "input_db": INPUT_DB,
            "source_table": SOURCE_TABLE,
            "raw_non_empty_comments": int(before),
            "comments_used_after_cleaning": int(len(df)),
            "n_topics": int(N_TOPICS),
            "max_features": int(MAX_FEATURES),
            "min_df": int(MIN_DF),
            "max_df": float(MAX_DF),
            "ngram_range": str(NGRAM_RANGE),
            "min_clean_words": int(MIN_CLEAN_WORDS),
            "top_terms_per_topic": int(TOP_TERMS_PER_TOPIC),
            "examples_per_topic": int(EXAMPLES_PER_TOPIC),
            "nmf_reconstruction_error": float(nmf.reconstruction_err_),
            **metrics,
        }
    ])

    print(f"Saving output DB: {OUTPUT_DB}")

    save_table(comment_topics_df, OUTPUT_DB, "nmf_all_comment_topics")
    save_table(topic_summary_df, OUTPUT_DB, "nmf_all_topics")
    save_table(topic_terms_df, OUTPUT_DB, "nmf_all_topic_terms")
    save_table(config_df, OUTPUT_DB, "nmf_all_model_config")

    print("\nDone.")
    print("Created tables:")
    print("- nmf_all_comment_topics")
    print("- nmf_all_topics")
    print("- nmf_all_topic_terms")
    print("- nmf_all_model_config")
    print("Output DB:", Path(OUTPUT_DB).resolve())


if __name__ == "__main__":
    main()
