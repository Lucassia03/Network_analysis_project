import re
import string
import sqlite3
from pathlib import Path

import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ==================================================
# CONFIG
# ==================================================

DB_PATH = "moltbook_final_NLI_sim_WITH_CLUSTER_AGENT_MAPPING.db"

OUTPUT_DB = "author_cluster_presence.db"
OUTPUT_TABLE = "author_cluster_presence"

COMMENTS_TABLE = "comments"

N_CLUSTERS = 7
RANDOM_STATE = 42


# ==================================================
# STOPWORDS
# ==================================================

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

stop_words = stop_words.union({
    "im", "ive", "id", "youre", "youve", "theyre", "thats", "dont",
    "didnt", "doesnt", "isnt", "arent", "wasnt", "werent", "cant",
    "couldnt", "wouldnt", "shouldnt", "ill", "theres", "heres",
    "amp", "nbsp", "lol", "yeah", "okay", "ok", "also", "really",
    "would", "could", "should", "get", "got", "one", "like", "just"
})


# ==================================================
# SQLITE CHECKS
# ==================================================

def check_database_exists(db_path: str) -> None:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path.resolve()}")


def check_table_exists(db_path: str, table_name: str) -> None:
    check_database_exists(db_path)

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
            tables = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                ORDER BY name;
                """
            ).fetchall()
            tables = [row[0] for row in tables]
            raise ValueError(
                f"Table '{table_name}' not found in {db_path}. "
                f"Available tables: {tables}"
            )
    finally:
        conn.close()


def check_required_columns(db_path: str, table_name: str, required_columns: set) -> None:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        columns = {row[1] for row in rows}

        missing = required_columns - columns
        if missing:
            raise KeyError(
                f"Missing required columns in table '{table_name}': {missing}. "
                f"Found columns: {sorted(columns)}"
            )
    finally:
        conn.close()


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):
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
        if tok not in stop_words and len(tok) > 2
    ]

    return " ".join(tokens)


# ==================================================
# MAIN
# ==================================================

def main():
    print("Checking input database...")
    check_table_exists(DB_PATH, COMMENTS_TABLE)
    check_required_columns(
        DB_PATH,
        COMMENTS_TABLE,
        required_columns={"id", "post_id", "author", "body"},
    )

    print("Loading comments...")
    conn = sqlite3.connect(DB_PATH)

    try:
        df = pd.read_sql_query(
            f"""
            SELECT
                id AS comment_id,
                post_id,
                author,
                body
            FROM {COMMENTS_TABLE}
            WHERE body IS NOT NULL
              AND TRIM(body) != '';
            """,
            conn,
        )
    finally:
        conn.close()

    print(f"Raw comments loaded: {len(df):,}")

    if df.empty:
        raise ValueError("No comments found in source table.")

    print("Normalizing authors...")
    df["author"] = (
        df["author"]
        .fillna("unknown_author")
        .astype(str)
        .str.strip()
    )

    df.loc[df["author"] == "", "author"] = "unknown_author"

    all_authors = (
        df.groupby("author")
        .agg(total_comments_in_dataset=("comment_id", "count"))
        .reset_index()
    )

    print(f"Unique authors: {len(all_authors):,}")

    print("Cleaning text...")
    df["cleaned_text"] = df["body"].fillna("").astype(str).apply(clean_text)

    before_cleaning = len(df)

    df_cluster = df[
        df["cleaned_text"].notna() &
        (df["cleaned_text"].str.strip() != "")
    ].copy()

    df_cluster = df_cluster[
        df_cluster["cleaned_text"].str.split().str.len() >= 4
    ].copy()

    print(f"Comments after cleaning: {len(df_cluster):,}")
    print(f"Comments removed: {before_cleaning - len(df_cluster):,}")

    if len(df_cluster) < N_CLUSTERS:
        raise ValueError(
            f"Not enough comments ({len(df_cluster)}) for {N_CLUSTERS} clusters."
        )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(df_cluster["cleaned_text"])

    print("TF-IDF matrix shape:", X.shape)

    print("Running KMeans clustering...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    df_cluster["cluster"] = kmeans.fit_predict(X).astype(int)

    print("\nCluster sizes:")
    print(df_cluster["cluster"].value_counts().sort_index().to_string())

    print("Building author-cluster presence table...")

    author_cluster_counts = (
        df_cluster
        .groupby(["author", "cluster"])
        .size()
        .reset_index(name="comments_in_cluster")
    )

    author_cluster_pivot = (
        author_cluster_counts
        .pivot(index="author", columns="cluster", values="comments_in_cluster")
        .fillna(0)
        .astype(int)
    )

    author_cluster_pivot.columns = [
        f"cluster_{col}" for col in author_cluster_pivot.columns
    ]

    author_cluster_pivot = author_cluster_pivot.reset_index()

    for i in range(N_CLUSTERS):
        col = f"cluster_{i}"
        if col not in author_cluster_pivot.columns:
            author_cluster_pivot[col] = 0

    cluster_cols = [f"cluster_{i}" for i in range(N_CLUSTERS)]

    author_cluster_pivot = author_cluster_pivot[
        ["author"] + cluster_cols
    ]

    author_presence = all_authors.merge(
        author_cluster_pivot,
        on="author",
        how="left",
    )

    author_presence[cluster_cols] = (
        author_presence[cluster_cols]
        .fillna(0)
        .astype(int)
    )

    author_presence["clustered_comments"] = author_presence[cluster_cols].sum(axis=1)

    author_presence["num_clusters_with_comments"] = (
        author_presence[cluster_cols] > 0
    ).sum(axis=1)

    author_presence["clusters_with_comments"] = author_presence.apply(
        lambda row: ", ".join(
            str(i) for i in range(N_CLUSTERS)
            if row[f"cluster_{i}"] > 0
        ),
        axis=1,
    )

    author_presence = author_presence[
        [
            "author",
            "total_comments_in_dataset",
            "clustered_comments",
            "num_clusters_with_comments",
            "clusters_with_comments",
        ] + cluster_cols
    ]

    author_presence = author_presence.sort_values(
        ["num_clusters_with_comments", "clustered_comments"],
        ascending=False,
    ).reset_index(drop=True)

    print(f"Saving output DB: {OUTPUT_DB}")

    conn_out = sqlite3.connect(OUTPUT_DB)
    try:
        author_presence.to_sql(
            OUTPUT_TABLE,
            conn_out,
            if_exists="replace",
            index=False,
        )
    finally:
        conn_out.close()

    print("\nDone.")
    print("Output DB:", Path(OUTPUT_DB).resolve())
    print("Created table:", OUTPUT_TABLE)
    print("\nPreview:")
    print(author_presence.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
