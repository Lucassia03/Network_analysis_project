import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DB_PATH = "moltbook.db"
OUTPUT_CSV = "comment_parent_similarity.csv"
OUTPUT_TABLE = "comment_parent_similarity"

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    c.id AS comment_id,
    c.post_id,
    c.depth,
    c.parent_comment_id,
    c.body AS comment_text,
    p.body AS post_text,
    pc.body AS parent_comment_text
FROM comments c
JOIN posts p
    ON c.post_id = p.id
LEFT JOIN comments pc
    ON c.parent_comment_id = pc.id
WHERE c.body IS NOT NULL
"""

df = pd.read_sql_query(query, conn)

df["comment_text"] = df["comment_text"].fillna("").str.strip()
df["post_text"] = df["post_text"].fillna("").str.strip()
df["parent_comment_text"] = df["parent_comment_text"].fillna("").str.strip()

df["parent_text"] = np.where(
    df["parent_comment_id"].notna() & (df["parent_comment_text"] != ""),
    df["parent_comment_text"],
    df["post_text"]
)

df = df[(df["comment_text"] != "") & (df["parent_text"] != "")].copy()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

unique_texts = pd.unique(
    pd.concat([df["comment_text"], df["parent_text"]], ignore_index=True)
)

embeddings = model.encode(
    unique_texts.tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

text_to_emb = dict(zip(unique_texts, embeddings))

comment_emb = np.vstack(df["comment_text"].map(text_to_emb).to_numpy())
parent_emb = np.vstack(df["parent_text"].map(text_to_emb).to_numpy())

df["sim_parent"] = np.sum(comment_emb * parent_emb, axis=1)

output_df = df[
    [
        "comment_id",
        "post_id",
        "depth",
        "parent_comment_id",
        "comment_text",
        "parent_text",
        "sim_parent",
    ]
].copy()

output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
output_df.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

conn.close()

print(f"Output CSV scritto in: {OUTPUT_CSV}")
print(f"Output tabella SQLite scritta in: {OUTPUT_TABLE}")
print(output_df.head())