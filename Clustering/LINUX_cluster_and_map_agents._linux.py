import sqlite3
import re
import string
import json

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans


# ==========================================================
# CONFIG
# ==========================================================

DB_PATH = "moltbook_final_NLI_sim.db"

COMMENTS_TABLE = "comments"
AGENTS_TABLE = "agent_predictions"

OUTPUT_COMMENT_CLUSTERS_TABLE = "comment_clusters"
OUTPUT_TARGET_CLUSTER_TABLE = "target_cluster_comments"
OUTPUT_MAPPING_TABLE = "comment_cluster_agent_mapping"
OUTPUT_TARGET_MAPPING_TABLE = "target_cluster_agent_mapping"
OUTPUT_CLUSTER_TERMS_TABLE = "comment_cluster_top_terms_mapping"

OUTPUT_COMMENT_CLUSTERS_CSV = "comment_clusters.csv"
OUTPUT_TARGET_CLUSTER_CSV = "target_cluster_comments.csv"
OUTPUT_MAPPING_CSV = "comment_cluster_agent_mapping.csv"
OUTPUT_TARGET_MAPPING_CSV = "target_cluster_agent_mapping.csv"
OUTPUT_CLUSTER_TERMS_CSV = "comment_cluster_top_terms_mapping.csv"

N_CLUSTERS = 7
RANDOM_STATE = 42

TARGET_CLUSTER_NAME = "great_question_skills_goals_rule_thumb"

TARGET_KEYWORDS = {
    "great question",
    "great",
    "skills",
    "might time",
    "goals",
    "rule thumb",
    "thumb",
    "good rule",
    "time",
    "question good",
    "good",
    "time let",
}


# ==========================================================
# 0. Stopwords
# ==========================================================

stop_words = set(ENGLISH_STOP_WORDS)

stop_words = stop_words.union({
    "im", "ive", "id", "youre", "youve", "theyre", "thats", "dont",
    "didnt", "doesnt", "isnt", "arent", "wasnt", "werent", "cant",
    "couldnt", "wouldnt", "shouldnt", "ill", "theres", "heres",
    "amp", "rt", "like", "just", "really"
})


# ==========================================================
# 1. Utility
# ==========================================================

def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = """
    SELECT name 
    FROM sqlite_master 
    WHERE type='table' AND name=?;
    """
    result = conn.execute(query, (table_name,)).fetchone()
    return result is not None


def load_comments(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    query = f"""
    SELECT 
        id AS comment_id,
        post_id,
        body
    FROM {table_name}
    WHERE body IS NOT NULL
      AND TRIM(body) != '';
    """

    return pd.read_sql_query(query, conn)


def clean_text(text: str) -> str:
    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # remove punctuation
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )

    # remove numbers
    text = re.sub(r"\d+", " ", text)

    # keep only english letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        tok for tok in text.split()
        if tok not in stop_words and len(tok) > 2
    ]

    return " ".join(tokens)


def find_best_matching_cluster(
    cluster_top_terms: dict,
    target_keywords: set
) -> tuple[int, dict]:
    normalized_targets = {
        keyword.strip().lower()
        for keyword in target_keywords
    }

    scores = {}

    for cluster_id, terms in cluster_top_terms.items():
        normalized_terms = {
            term.strip().lower()
            for term in terms
        }

        exact_overlap = normalized_terms.intersection(normalized_targets)

        partial_hits = 0

        for target in normalized_targets:
            for term in normalized_terms:
                if target in term or term in target:
                    partial_hits += 1

        score = len(exact_overlap) * 10 + partial_hits

        scores[cluster_id] = {
            "score": score,
            "exact_overlap": sorted(exact_overlap),
            "top_terms": terms,
        }

    best_cluster = max(scores, key=lambda cluster_id: scores[cluster_id]["score"])

    return best_cluster, scores


def save_table_and_csv(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table_name: str,
    csv_path: str
) -> None:
    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    df.to_csv(
        csv_path,
        index=False
    )


# ==========================================================
# 2. Main pipeline
# ==========================================================

def main():
    print("Opening database:", DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    try:
        if not table_exists(conn, COMMENTS_TABLE):
            raise ValueError(f"Table '{COMMENTS_TABLE}' not found in database.")

        if not table_exists(conn, AGENTS_TABLE):
            raise ValueError(
                f"Table '{AGENTS_TABLE}' not found in database. "
                "Run agent prediction first."
            )

        # --------------------------------------------------
        # Load comments
        # --------------------------------------------------

        print("Loading comments...")

        df = load_comments(conn, COMMENTS_TABLE)

        print("Initial comments:", len(df))

        # --------------------------------------------------
        # Clean text
        # --------------------------------------------------

        print("Cleaning text...")

        df["cleaned_text"] = df["body"].astype(str).apply(clean_text)

        df = df[df["cleaned_text"].notna()].copy()
        df = df[df["cleaned_text"].str.strip() != ""].copy()

        before_len = len(df)

        df = df[
            df["cleaned_text"].str.split().str.len() >= 4
        ].copy()

        after_len = len(df)

        print("Comments after cleaning:", after_len)
        print("Comments removed because too short:", before_len - after_len)

        if after_len == 0:
            raise ValueError("No comments left after cleaning.")

        if after_len < N_CLUSTERS:
            raise ValueError(
                f"Not enough comments after cleaning: {after_len}. "
                f"n_clusters={N_CLUSTERS}."
            )

        # --------------------------------------------------
        # Vectorize
        # --------------------------------------------------

        print("Vectorizing...")

        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=3,
            max_df=0.85,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        X = vectorizer.fit_transform(df["cleaned_text"])

        print("Matrix shape:", X.shape)

        # --------------------------------------------------
        # Cluster
        # --------------------------------------------------

        print("Clustering...")

        kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init=20
        )

        df["cluster"] = kmeans.fit_predict(X)

        print("\nCluster sizes:")
        print(df["cluster"].value_counts().sort_index())

        # --------------------------------------------------
        # Top terms
        # --------------------------------------------------

        feature_names = np.array(vectorizer.get_feature_names_out())
        centers = kmeans.cluster_centers_

        cluster_top_terms = {}
        cluster_terms_records = []

        print("\nTop words per cluster:")

        for cluster_id in range(N_CLUSTERS):
            top_idx = centers[cluster_id].argsort()[::-1][:20]
            top_terms = feature_names[top_idx].tolist()

            cluster_top_terms[cluster_id] = top_terms

            print(f"Cluster {cluster_id}: {', '.join(top_terms[:12])}")

            for rank, term in enumerate(top_terms, start=1):
                cluster_terms_records.append({
                    "cluster": cluster_id,
                    "rank": rank,
                    "term": term
                })

        cluster_terms_df = pd.DataFrame(cluster_terms_records)

        # --------------------------------------------------
        # Find target cluster
        # --------------------------------------------------

        target_cluster, cluster_match_scores = find_best_matching_cluster(
            cluster_top_terms,
            TARGET_KEYWORDS
        )

        print("\nTarget semantic cluster identified:")
        print("cluster_id:", target_cluster)
        print("top_terms:", cluster_top_terms[target_cluster])
        print(
            "match_info:",
            json.dumps(
                cluster_match_scores[target_cluster],
                ensure_ascii=False,
                indent=2
            )
        )

        df["semantic_cluster_name"] = np.where(
            df["cluster"] == target_cluster,
            TARGET_CLUSTER_NAME,
            None
        )

        df["is_target_cluster"] = (
            df["cluster"] == target_cluster
        ).astype(int)

        comment_clusters = df[
            [
                "comment_id",
                "post_id",
                "body",
                "cleaned_text",
                "cluster",
                "semantic_cluster_name",
                "is_target_cluster",
            ]
        ].copy()

        # --------------------------------------------------
        # Save clusters
        # --------------------------------------------------

        print("\nSaving comment cluster tables...")

        save_table_and_csv(
            conn=conn,
            df=comment_clusters,
            table_name=OUTPUT_COMMENT_CLUSTERS_TABLE,
            csv_path=OUTPUT_COMMENT_CLUSTERS_CSV
        )

        target_only = comment_clusters[
            comment_clusters["is_target_cluster"] == 1
        ].copy()

        save_table_and_csv(
            conn=conn,
            df=target_only,
            table_name=OUTPUT_TARGET_CLUSTER_TABLE,
            csv_path=OUTPUT_TARGET_CLUSTER_CSV
        )

        save_table_and_csv(
            conn=conn,
            df=cluster_terms_df,
            table_name=OUTPUT_CLUSTER_TERMS_TABLE,
            csv_path=OUTPUT_CLUSTER_TERMS_CSV
        )

        # --------------------------------------------------
        # Join with agent predictions
        # --------------------------------------------------

        print("Joining with agent predictions...")

        agents_df = pd.read_sql_query(
            f"SELECT * FROM {AGENTS_TABLE}",
            conn
        )

        if "id" in agents_df.columns and "comment_id" not in agents_df.columns:
            agents_df = agents_df.rename(columns={"id": "comment_id"})

        if "comment_id" not in agents_df.columns:
            raise KeyError(
                "Missing required column 'comment_id' in agent_predictions."
            )

        mapping_df = comment_clusters.merge(
            agents_df,
            on="comment_id",
            how="left",
            suffixes=("", "_agent")
        )

        save_table_and_csv(
            conn=conn,
            df=mapping_df,
            table_name=OUTPUT_MAPPING_TABLE,
            csv_path=OUTPUT_MAPPING_CSV
        )

        target_mapping_df = mapping_df[
            mapping_df["is_target_cluster"] == 1
        ].copy()

        save_table_and_csv(
            conn=conn,
            df=target_mapping_df,
            table_name=OUTPUT_TARGET_MAPPING_TABLE,
            csv_path=OUTPUT_TARGET_MAPPING_CSV
        )

    finally:
        conn.close()

    print("\nDone.")
    print("Updated DB:", DB_PATH)
    print("Created SQL tables:")
    print("-", OUTPUT_COMMENT_CLUSTERS_TABLE)
    print("-", OUTPUT_TARGET_CLUSTER_TABLE)
    print("-", OUTPUT_CLUSTER_TERMS_TABLE)
    print("-", OUTPUT_MAPPING_TABLE)
    print("-", OUTPUT_TARGET_MAPPING_TABLE)
    print("\nCreated CSV files:")
    print("-", OUTPUT_COMMENT_CLUSTERS_CSV)
    print("-", OUTPUT_TARGET_CLUSTER_CSV)
    print("-", OUTPUT_CLUSTER_TERMS_CSV)
    print("-", OUTPUT_MAPPING_CSV)
    print("-", OUTPUT_TARGET_MAPPING_CSV)


if __name__ == "__main__":
    main()