import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=False
)

# Load NLI model
classifier = pipeline("text-classification", model="facebook/bart-large-mnli")

# Load database
db_path = "moltbook.db"
conn = sqlite3.connect(db_path)

posts = pd.read_sql("SELECT * FROM posts;", conn)
comments = pd.read_sql("SELECT * FROM comments;", conn)

# Merge posts and comments
df = comments.merge(
    posts,
    left_on="post_id",
    right_on="id",
    suffixes=("_comment", "_post")
)

# Clean text
df["body_comment"] = df["body_comment"].fillna("").str.lower()
df["body_post"] = df["body_post"].fillna("").str.lower()

# Filter direct replies only
df = df[df["depth"] == 1].copy()

# Encode text
print("Encoding comments...")
comment_embeddings = model.encode(
    df["body_comment"].tolist(),
    show_progress_bar=True
)

print("Encoding posts...")
post_embeddings = model.encode(
    df["body_post"].tolist(),
    show_progress_bar=True
)
# Compute similarity
df["similarity"] = [
    cosine_similarity([p], [c])[0][0]
    for p, c in zip(post_embeddings, comment_embeddings)
]

print(df[["body_post", "body_comment", "similarity"]].head())
# Keep only relevant rows

