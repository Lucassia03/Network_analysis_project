import sqlite3
import re
import string
import json
from pathlib import Path

import pandas as pd
import numpy as np
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

COMMENTS_DB = "moltbook_final.db"
AGENTS_DB = "agent_predictions_from_json.db"
OUTPUT_DB = "cluster_agent_mapping.db"

COMMENTS_TABLE = "comments"
AGENTS_TABLE = "agent_predictions"

N_CLUSTERS = 7
RANDOM_STATE = 42


# --------------------------------------------------
# Keywords per stimare sicofantia / adulazione
# --------------------------------------------------
# Nota: misura linguaggio approvativo/adulatorio.
# Non è una vera tone analysis contestuale.

SYCOPHANCY_KEYWORDS = {
    "great question",
    "good question",
    "excellent question",
    "great point",
    "good point",
    "excellent point",
    "well said",
    "exactly",
    "totally agree",
    "completely agree",
    "strongly agree",
    "i agree",
    "agree",
    "love this",
    "i love this",
    "this is great",
    "this is good",
    "this is helpful",
    "very helpful",
    "super helpful",
    "really helpful",
    "amazing",
    "brilliant",
    "insightful",
    "nice",
    "good",
    "great",
    "excellent",
    "smart",
    "valuable",
    "interesting",
    "thanks",
    "thank you",
    "appreciate",
    "makes sense",
    "you are right",
    "youre right",
    "right",
    "correct",
    "true",
    "spot on",
    "absolutely",
    "perfect",
    "nailed it",
    "couldnt agree more",
    "could not agree more",
}


# --------------------------------------------------
# Nomi opzionali dei 7 cluster già interpretati
# --------------------------------------------------
# Attenzione: questi nomi sono validi solo se il clustering
# produce cluster simili a quelli che avevi già ottenuto.

CLUSTER_NAMES = {
    0: "onchain_coinflip_promo",
    1: "distributed_memory_storage_latency",
    2: "agent_governance_trust_security",
    3: "general_memory_identity_systems_discussion",
    4: "locivault_private_encrypted_memory",
    5: "distributed_consistency_gossip_sync",
    6: "resume_career_narrative_skills",
}


# --------------------------------------------------
# 0) Stopwords
# --------------------------------------------------

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
    "couldnt", "wouldnt", "shouldnt", "ill", "theres", "heres"
})


# --------------------------------------------------
# 1) SQLite utilities
# --------------------------------------------------

def check_table_exists(db_path: str, table_name: str) -> None:
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?;
        """

        result = conn.execute(query, (table_name,)).fetchone()

        if result is None:
            raise ValueError(
                f"Table '{table_name}' not found in database: {db_path}"
            )

    finally:
        conn.close()


def load_comments(db_path: str, table_name: str) -> pd.DataFrame:
    check_table_exists(db_path, table_name)

    conn = sqlite3.connect(db_path)

    try:
        query = f"""
        SELECT id AS comment_id, post_id, body
        FROM {table_name}
        WHERE body IS NOT NULL
          AND TRIM(body) != '';
        """

        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    return df


def save_table(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    conn = sqlite3.connect(db_path)

    try:
        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

    finally:
        conn.close()


# --------------------------------------------------
# 2) Text preprocessing
# --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Pulizia forte usata per TF-IDF e clustering.
    """

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


def normalize_for_keyword_matching(text: str) -> str:
    """
    Pulizia più leggera per cercare frasi tipo:
    'great question', 'thank you', 'well said'.
    """

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )

    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# 3) Sycophancy scoring
# --------------------------------------------------

def compute_sycophancy_score_for_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    original_text_col: str,
    cluster_top_terms: dict,
    keywords: set
) -> pd.DataFrame:
    """
    Calcola uno score keyword-based di possibile sicofantia per cluster.

    Segnali:
    - quota di commenti del cluster con almeno una keyword;
    - keyword hits ogni 100 commenti;
    - overlap tra keyword e top terms del cluster.
    """

    normalized_keywords = [
        normalize_for_keyword_matching(kw)
        for kw in keywords
    ]
    normalized_keywords = sorted(set(normalized_keywords))

    rows = []

    for cluster_id, group in df.groupby(cluster_col):
        original_texts = (
            group[original_text_col]
            .fillna("")
            .astype(str)
            .apply(normalize_for_keyword_matching)
        )

        n_comments = len(group)

        keyword_comment_hits = 0
        total_keyword_hits = 0
        keyword_hit_breakdown = {}

        for text in original_texts:
            comment_has_hit = False

            for kw in normalized_keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                hits = len(re.findall(pattern, text))

                if hits > 0:
                    comment_has_hit = True
                    total_keyword_hits += hits
                    keyword_hit_breakdown[kw] = (
                        keyword_hit_breakdown.get(kw, 0) + hits
                    )

            if comment_has_hit:
                keyword_comment_hits += 1

        top_terms = [
            normalize_for_keyword_matching(term)
            for term in cluster_top_terms.get(cluster_id, [])
        ]

        exact_top_term_overlap = sorted(
            set(top_terms).intersection(set(normalized_keywords))
        )

        partial_top_term_overlap = []

        for kw in normalized_keywords:
            for term in top_terms:
                if kw in term or term in kw:
                    partial_top_term_overlap.append(term)

        partial_top_term_overlap = sorted(set(partial_top_term_overlap))

        comment_hit_rate = keyword_comment_hits / n_comments if n_comments else 0

        keyword_hits_per_100_comments = (
            total_keyword_hits / n_comments * 100
            if n_comments else 0
        )

        top_term_overlap_score = (
            len(exact_top_term_overlap) * 2
            + len(partial_top_term_overlap)
        )

        sycophancy_score = (
            comment_hit_rate * 100
            + keyword_hits_per_100_comments
            + top_term_overlap_score
        )

        top_keywords_in_cluster = sorted(
            keyword_hit_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        example_hits = []

        for _, row in group.iterrows():
            normalized_text = normalize_for_keyword_matching(
                row[original_text_col]
            )

            matched_keywords = []

            for kw in normalized_keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"

                if re.search(pattern, normalized_text):
                    matched_keywords.append(kw)

            if matched_keywords:
                example_hits.append({
                    "comment_id": row["comment_id"],
                    "post_id": row["post_id"],
                    "matched_keywords": matched_keywords[:5],
                    "body": str(row[original_text_col])[:500],
                })

            if len(example_hits) >= 5:
                break

        rows.append({
            "cluster": int(cluster_id),
            "cluster_name": CLUSTER_NAMES.get(
                int(cluster_id),
                f"cluster_{cluster_id}"
            ),
            "n_comments": int(n_comments),
            "keyword_comment_hits": int(keyword_comment_hits),
            "comment_hit_rate": float(comment_hit_rate),
            "comment_hit_rate_pct": float(comment_hit_rate * 100),
            "total_keyword_hits": int(total_keyword_hits),
            "keyword_hits_per_100_comments": float(keyword_hits_per_100_comments),
            "top_term_overlap_score": float(top_term_overlap_score),
            "sycophancy_score": float(sycophancy_score),
            "exact_top_term_overlap": json.dumps(
                exact_top_term_overlap,
                ensure_ascii=False
            ),
            "partial_top_term_overlap": json.dumps(
                partial_top_term_overlap,
                ensure_ascii=False
            ),
            "top_keywords_in_cluster": json.dumps(
                top_keywords_in_cluster,
                ensure_ascii=False
            ),
            "example_hits": json.dumps(
                example_hits,
                ensure_ascii=False
            ),
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "sycophancy_score",
        ascending=False
    ).reset_index(drop=True)

    result["sycophancy_rank"] = range(1, len(result) + 1)

    return result


def add_comment_level_sycophancy_features(
    df: pd.DataFrame,
    text_col: str,
    keywords: set
) -> pd.DataFrame:
    """
    Aggiunge feature a livello di singolo commento:
    - sycophancy_keyword_count
    - sycophancy_matched_keywords
    - has_sycophancy_keyword
    """

    normalized_keywords = [
        normalize_for_keyword_matching(kw)
        for kw in keywords
    ]
    normalized_keywords = sorted(set(normalized_keywords))

    counts = []
    matched_all = []

    for text in df[text_col].fillna("").astype(str):
        normalized_text = normalize_for_keyword_matching(text)

        total_hits = 0
        matched_keywords = []

        for kw in normalized_keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            hits = len(re.findall(pattern, normalized_text))

            if hits > 0:
                total_hits += hits
                matched_keywords.append(kw)

        counts.append(total_hits)
        matched_all.append(json.dumps(matched_keywords, ensure_ascii=False))

    df = df.copy()

    df["sycophancy_keyword_count"] = counts
    df["sycophancy_matched_keywords"] = matched_all
    df["has_sycophancy_keyword"] = (
        df["sycophancy_keyword_count"] > 0
    ).astype(int)

    return df


# --------------------------------------------------
# 4) Main pipeline
# --------------------------------------------------

def main():
    print("Loading comments...")

    df = load_comments(COMMENTS_DB, COMMENTS_TABLE)

    print("Initial comments:", len(df))

    print("Cleaning text...")

    df["cleaned_text"] = df["body"].astype(str).apply(clean_text)

    df = df[df["cleaned_text"].notna()]
    df = df[df["cleaned_text"].str.strip() != ""]

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
            f"Not enough comments ({after_len}) for {N_CLUSTERS} clusters."
        )

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

    print("Clustering...")

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20
    )

    df["cluster"] = kmeans.fit_predict(X)

    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())

    feature_names = np.array(vectorizer.get_feature_names_out())
    centers = kmeans.cluster_centers_

    cluster_top_terms = {}

    print("\nTop words per cluster:")

    for cluster_id in range(N_CLUSTERS):
        top_idx = centers[cluster_id].argsort()[::-1][:20]
        top_terms = feature_names[top_idx].tolist()

        cluster_top_terms[cluster_id] = top_terms

        print(f"Cluster {cluster_id}: {', '.join(top_terms)}")

    print("\nAdding comment-level sycophancy keyword features...")

    df = add_comment_level_sycophancy_features(
        df=df,
        text_col="body",
        keywords=SYCOPHANCY_KEYWORDS
    )

    print("\nComputing sycophancy score per cluster...")

    sycophancy_cluster_scores = compute_sycophancy_score_for_clusters(
        df=df,
        cluster_col="cluster",
        original_text_col="body",
        cluster_top_terms=cluster_top_terms,
        keywords=SYCOPHANCY_KEYWORDS
    )

    print("\nSycophancy cluster ranking:")
    print(
        sycophancy_cluster_scores[
            [
                "sycophancy_rank",
                "cluster",
                "cluster_name",
                "n_comments",
                "comment_hit_rate_pct",
                "keyword_hits_per_100_comments",
                "top_term_overlap_score",
                "sycophancy_score",
                "top_keywords_in_cluster",
            ]
        ].to_string(index=False)
    )

    most_sycophantic_cluster = int(
        sycophancy_cluster_scores.iloc[0]["cluster"]
    )

    most_sycophantic_cluster_name = sycophancy_cluster_scores.iloc[0][
        "cluster_name"
    ]

    print("\nMost sycophantic cluster identified:")
    print("cluster_id:", most_sycophantic_cluster)
    print("cluster_name:", most_sycophantic_cluster_name)

    df["cluster_name"] = df["cluster"].map(
        lambda x: CLUSTER_NAMES.get(int(x), f"cluster_{x}")
    )

    df["is_most_sycophantic_cluster"] = (
        df["cluster"] == most_sycophantic_cluster
    ).astype(int)

    comment_clusters = df[
        [
            "comment_id",
            "post_id",
            "body",
            "cleaned_text",
            "cluster",
            "cluster_name",
            "sycophancy_keyword_count",
            "sycophancy_matched_keywords",
            "has_sycophancy_keyword",
            "is_most_sycophantic_cluster",
        ]
    ].copy()

    print("\nSaving comment_clusters...")

    save_table(
        comment_clusters,
        OUTPUT_DB,
        "comment_clusters"
    )

    print("Saving sycophancy_cluster_scores...")

    save_table(
        sycophancy_cluster_scores,
        OUTPUT_DB,
        "sycophancy_cluster_scores"
    )

    most_sycophantic_comments = comment_clusters[
        comment_clusters["is_most_sycophantic_cluster"] == 1
    ].copy()

    print("Saving most_sycophantic_cluster_comments...")

    save_table(
        most_sycophantic_comments,
        OUTPUT_DB,
        "most_sycophantic_cluster_comments"
    )

    keyword_hit_comments = comment_clusters[
        comment_clusters["has_sycophancy_keyword"] == 1
    ].copy()

    print("Saving comments_with_sycophancy_keywords...")

    save_table(
        keyword_hit_comments,
        OUTPUT_DB,
        "comments_with_sycophancy_keywords"
    )

    print("Joining with agent predictions...")

    check_table_exists(AGENTS_DB, AGENTS_TABLE)

    conn_out = sqlite3.connect(OUTPUT_DB)
    conn_agents = sqlite3.connect(AGENTS_DB)

    try:
        agents_df = pd.read_sql_query(
            f"SELECT * FROM {AGENTS_TABLE}",
            conn_agents
        )

        if "id" in agents_df.columns and "comment_id" not in agents_df.columns:
            agents_df = agents_df.rename(columns={"id": "comment_id"})

        missing = {"comment_id"} - set(agents_df.columns)

        if missing:
            raise KeyError(
                f"Missing required columns in agent predictions table: {missing}"
            )

        mapping_df = comment_clusters.merge(
            agents_df,
            on="comment_id",
            how="left"
        )

        mapping_df.to_sql(
            "comment_cluster_agent_mapping",
            conn_out,
            if_exists="replace",
            index=False
        )

        most_sycophantic_mapping_df = mapping_df[
            mapping_df["is_most_sycophantic_cluster"] == 1
        ].copy()

        most_sycophantic_mapping_df.to_sql(
            "most_sycophantic_cluster_agent_mapping",
            conn_out,
            if_exists="replace",
            index=False
        )

        keyword_hit_mapping_df = mapping_df[
            mapping_df["has_sycophancy_keyword"] == 1
        ].copy()

        keyword_hit_mapping_df.to_sql(
            "sycophancy_keyword_agent_mapping",
            conn_out,
            if_exists="replace",
            index=False
        )

    finally:
        conn_out.close()
        conn_agents.close()

    print("\nDone.")
    print("Output DB:", OUTPUT_DB)
    print("Created tables:")
    print("- comment_clusters")
    print("- sycophancy_cluster_scores")
    print("- most_sycophantic_cluster_comments")
    print("- comments_with_sycophancy_keywords")
    print("- comment_cluster_agent_mapping")
    print("- most_sycophantic_cluster_agent_mapping")
    print("- sycophancy_keyword_agent_mapping")


if __name__ == "__main__":
    main()
