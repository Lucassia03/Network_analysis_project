import sqlite3
from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np 

model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
db_path = "moltbook.db"
conn = sqlite3.connect(db_path)

posts = pd.read_sql("SELECT * FROM posts;", conn)
comments = pd.read_sql("SELECT * FROM comments;", conn)
df = comments.merge(
    posts,
    left_on="post_id",
    right_on="id",
    suffixes=("_comment", "_post")
)


pairs = df[["body_post", "body_comment"]].values.tolist()
predictions = model.predict(pairs, show_progress_bar=True)
print(predictions[:10])

labels = ["contradiction", "neutral", "entailment"]
df["nli_prediction"] = [labels[i] for i in np.argmax(predictions, axis=1)]
print(df["nli_prediction"].head())
# Connect to database
conn = sqlite3.connect("moltbook.db")
cursor = conn.cursor()

# Create table with 3 scores + label
cursor.execute("""
    CREATE TABLE IF NOT EXISTS nli_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        comment_id INTEGER,
        post_id INTEGER,
        contradiction_score REAL,
        neutral_score REAL,
        entailment_score REAL,
        prediction TEXT,
        FOREIGN KEY(comment_id) REFERENCES comments(id),
        FOREIGN KEY(post_id) REFERENCES posts(id)
    )
""")

# Insert everything coherently
for i, (_, row) in enumerate(df.iterrows()):
    scores = predictions[i]
    
    cursor.execute("""
        INSERT INTO nli_predictions (
            comment_id, post_id,
            contradiction_score, neutral_score, entailment_score,
            prediction
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["id_comment"],
        row["post_id"],
        float(scores[0]),
        float(scores[1]),
        float(scores[2]),
        row["nli_prediction"]
    ))

conn.commit()
conn.close()

print("Predictions (scores + label) saved to moltbook.db")