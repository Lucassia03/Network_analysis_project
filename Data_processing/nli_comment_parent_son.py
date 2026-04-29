import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder

DB_PATH = "moltbook.db"
OUTPUT_TABLE = "nli_comment_parent_predictions"
OUTPUT_CSV = "nli_comment_parent_predictions.csv"

# =========================================================
# 1. Load model
# =========================================================
model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# =========================================================
# 2. Connect to database
# =========================================================
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# =========================================================
# 3. Read comment-comment pairs
# =========================================================
# ASSUNZIONE:
# - comments.id = id del commento
# - comments.parent_comment_id = id del commento padre
# - comments.body = testo commento
#
# Se il tuo schema usa un nome diverso (es. parent_id),
# sostituisci c.parent_comment_id con il nome corretto.

query = """
SELECT
    c.id AS comment_id,
    c.post_id AS post_id,
    c.parent_comment_id AS parent_comment_id,
    pc.id AS actual_parent_comment_id,
    pc.body AS parent_text,
    c.body AS comment_text,
    c.depth AS depth
FROM comments c
JOIN comments pc
    ON c.parent_comment_id = pc.id
WHERE c.body IS NOT NULL
  AND pc.body IS NOT NULL
"""

df = pd.read_sql_query(query, conn)

# =========================================================
# 4. Clean text
# =========================================================
df["parent_text"] = df["parent_text"].fillna("").astype(str).str.strip()
df["comment_text"] = df["comment_text"].fillna("").astype(str).str.strip()

df = df[(df["parent_text"] != "") & (df["comment_text"] != "")].copy()

# =========================================================
# 5. Build NLI pairs: (parent_comment, child_comment)
# =========================================================
pairs = list(zip(df["parent_text"].tolist(), df["comment_text"].tolist()))

# =========================================================
# 6. Predict NLI scores
# =========================================================
predictions = model.predict(pairs, show_progress_bar=True)

labels = ["contradiction", "neutral", "entailment"]
df["contradiction_score"] = predictions[:, 0]
df["neutral_score"] = predictions[:, 1]
df["entailment_score"] = predictions[:, 2]
df["prediction"] = [labels[i] for i in np.argmax(predictions, axis=1)]

# =========================================================
# 7. Keep final output columns
# =========================================================
output_df = df[
    [
        "comment_id",
        "post_id",
        "parent_comment_id",
        "actual_parent_comment_id",
        "depth",
        "parent_text",
        "comment_text",
        "contradiction_score",
        "neutral_score",
        "entailment_score",
        "prediction",
    ]
].copy()

# =========================================================
# 8. Save to CSV
# =========================================================
output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

# =========================================================
# 9. Save to SQLite
# =========================================================
cursor.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")

cursor.execute(f"""
    CREATE TABLE {OUTPUT_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        comment_id INTEGER,
        post_id INTEGER,
        parent_comment_id INTEGER,
        actual_parent_comment_id INTEGER,
        depth INTEGER,
        parent_text TEXT,
        comment_text TEXT,
        contradiction_score REAL,
        neutral_score REAL,
        entailment_score REAL,
        prediction TEXT,
        FOREIGN KEY(comment_id) REFERENCES comments(id),
        FOREIGN KEY(post_id) REFERENCES posts(id),
        FOREIGN KEY(parent_comment_id) REFERENCES comments(id)
    )
""")

output_df.to_sql(OUTPUT_TABLE, conn, if_exists="append", index=False)

conn.commit()
conn.close()

print(f"NLI comment-parent salvato in tabella SQLite: {OUTPUT_TABLE}")
print(f"NLI comment-parent salvato in CSV: {OUTPUT_CSV}")
print(output_df.head())