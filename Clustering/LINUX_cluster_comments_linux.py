import sqlite3
import re
import string
import pandas as pd
import numpy as np
import matplotlib

# Backend non interattivo per server Linux
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from matplotlib.lines import Line2D


# ==========================================================
# 1. Settings
# ==========================================================

db_path = "moltbook_final_NLI_sim.db"

output_csv = "comment_lexical_clusters.csv"
output_top_terms_csv = "comment_cluster_top_terms.csv"
output_examples_csv = "comment_cluster_examples.csv"
output_png = "comment_clusters_2d.png"

sql_output_table = "comment_lexical_clusters"

n_clusters = 7
max_features = 5000
min_df = 3
max_df = 0.85
random_state = 42


# ==========================================================
# 2. Stopwords
# ==========================================================

stop_words = set(ENGLISH_STOP_WORDS)

stop_words = stop_words.union({
    "im", "ive", "id", "youre", "youve", "theyre", "thats", "dont",
    "didnt", "doesnt", "isnt", "arent", "wasnt", "werent", "cant",
    "couldnt", "wouldnt", "shouldnt", "ill", "theres", "heres",
    "amp", "rt", "like", "just", "really"
})


# ==========================================================
# 3. Load comments from SQLite
# ==========================================================

print("Loading comments from database...")

conn = sqlite3.connect(db_path)

comments_df = pd.read_sql(
    """
    SELECT 
        id AS comment_id,
        post_id,
        body
    FROM comments
    WHERE body IS NOT NULL
      AND TRIM(body) != '';
    """,
    conn
)

print("Number of initial comments:", len(comments_df))


# ==========================================================
# 4. Text preprocessing
# ==========================================================

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

    # remove stopwords and short words
    tokens = [
        tok for tok in text.split()
        if tok not in stop_words and len(tok) > 2
    ]

    return " ".join(tokens)


print("Cleaning text...")

comments_df["cleaned_text"] = comments_df["body"].astype(str).apply(clean_text)


# ==========================================================
# 5. Remove empty / too short comments after cleaning
# ==========================================================

comments_df = comments_df[comments_df["cleaned_text"].notna()].copy()
comments_df = comments_df[comments_df["cleaned_text"].str.strip() != ""].copy()

before_len = len(comments_df)

comments_df = comments_df[
    comments_df["cleaned_text"].str.split().str.len() >= 4
].copy()

after_len = len(comments_df)

print("Comments after cleaning:", after_len)
print("Comments removed because too short:", before_len - after_len)

if after_len == 0:
    raise ValueError("No comments left after cleaning.")


# ==========================================================
# 6. Vectorization
# ==========================================================

print("Vectorizing with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=max_features,
    min_df=min_df,
    max_df=max_df,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(comments_df["cleaned_text"])

print("Shape matrix:", X.shape)


# ==========================================================
# 7. Clustering
# ==========================================================

print("Running KMeans clustering...")

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=random_state,
    n_init=20
)

comments_df["cluster"] = kmeans.fit_predict(X)

print("\nCluster dimensions:")
print(comments_df["cluster"].value_counts().sort_index())


# ==========================================================
# 8. Fixed colors for clusters
# ==========================================================

cluster_colors = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#7f7f7f"
}

comments_df["color"] = comments_df["cluster"].map(cluster_colors)


# ==========================================================
# 9. Top terms per cluster
# ==========================================================

print("\nTOP WORDS PER CLUSTER")

feature_names = np.array(vectorizer.get_feature_names_out())
centers = kmeans.cluster_centers_

top_terms_records = []

for cluster_id in range(n_clusters):
    top_idx = centers[cluster_id].argsort()[::-1][:20]
    top_terms = feature_names[top_idx]

    print(f"\nCluster {cluster_id} ({cluster_colors[cluster_id]})")
    print(", ".join(top_terms[:12]))

    for rank, term in enumerate(top_terms, start=1):
        top_terms_records.append({
            "cluster": cluster_id,
            "color": cluster_colors[cluster_id],
            "rank": rank,
            "term": term
        })

top_terms_df = pd.DataFrame(top_terms_records)


# ==========================================================
# 10. Example comments per cluster
# ==========================================================

print("\nEXAMPLES OF COMMENTS PER CLUSTER")

examples_records = []

for cluster_id in range(n_clusters):
    print(f"\n=== Cluster {cluster_id} ({cluster_colors[cluster_id]}) ===")

    sample_comments = (
        comments_df[comments_df["cluster"] == cluster_id]
        [["comment_id", "post_id", "body"]]
        .head(10)
    )

    for rank, row in enumerate(sample_comments.itertuples(index=False), start=1):
        print(f"{rank}. {str(row.body)[:250]}")

        examples_records.append({
            "cluster": cluster_id,
            "color": cluster_colors[cluster_id],
            "rank": rank,
            "comment_id": row.comment_id,
            "post_id": row.post_id,
            "body": row.body
        })

examples_df = pd.DataFrame(examples_records)


# ==========================================================
# 11. 2D Projection with SVD
# ==========================================================

print("Computing 2D projection with SVD...")

svd_2d = TruncatedSVD(
    n_components=2,
    random_state=random_state
)

X_2d = svd_2d.fit_transform(X)

comments_df["svd_x"] = X_2d[:, 0]
comments_df["svd_y"] = X_2d[:, 1]


# ==========================================================
# 12. Plot 2D clusters
# ==========================================================

print("Saving 2D plot...")

legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label=f"Cluster {cid}",
        markerfacecolor=cluster_colors[cid],
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=8
    )
    for cid in sorted(cluster_colors)
]

plt.figure(figsize=(13, 9), dpi=180)

for cid in range(n_clusters):
    mask = comments_df["cluster"] == cid

    plt.scatter(
        comments_df.loc[mask, "svd_x"],
        comments_df.loc[mask, "svd_y"],
        s=16,
        alpha=0.72,
        c=cluster_colors[cid],
        label=f"Cluster {cid}"
    )

plt.title("Lexical Clusters of Comments - 2D (TF-IDF + KMeans)", fontsize=16)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(
    handles=legend_handles,
    title="Clusters",
    loc="best",
    frameon=True
)
plt.tight_layout()

plt.savefig(output_png, bbox_inches="tight")
plt.close()


# ==========================================================
# 13. Save outputs
# ==========================================================

print("Saving CSV outputs...")

comments_df.to_csv(output_csv, index=False)
top_terms_df.to_csv(output_top_terms_csv, index=False)
examples_df.to_csv(output_examples_csv, index=False)

print("Saving SQL table...")

comments_df.to_sql(
    sql_output_table,
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Done.")
print(f"Saved comments with clusters: {output_csv}")
print(f"Saved top terms: {output_top_terms_csv}")
print(f"Saved examples: {output_examples_csv}")
print(f"Saved plot: {output_png}")
print(f"Saved SQL table: {sql_output_table}")