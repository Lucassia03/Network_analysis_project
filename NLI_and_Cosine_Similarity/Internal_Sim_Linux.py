# ==========================================================
# Internal cosine similarity for both posts and comments
# Standalone Linux/HPC version for moltbook_final.db
# ==========================================================

import re
import sqlite3
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ==========================================================
# CONFIG
# ==========================================================

DB_PATH = "moltbook_final.db"
OUTPUT_TABLE = "text_internal_similarity"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================================
# 1. Function to split text into sentences
# ==========================================================

def split_into_sentences(text):
    """
    Splits a text into sentences using punctuation marks.
    Returns a list of non-empty sentences.
    """
    text = str(text).strip()

    sentences = re.split(r"[.!?]+", text)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() != ""
    ]

    return sentences


# ==========================================================
# 2. Function to compute internal cosine similarity
# ==========================================================

def compute_internal_similarity(text, embedding_model):
    """
    Computes the average cosine similarity among all sentence pairs
    inside the same text.

    If the text has fewer than 2 sentences, returns NaN.

    Higher value:
        sentences inside the same text are semantically similar.

    Lower value:
        sentences inside the same text are more semantically diverse.
    """
    sentences = split_into_sentences(text)

    if len(sentences) < 2:
        return np.nan

    embeddings = embedding_model.encode(
        sentences,
        show_progress_bar=False
    )

    sim_matrix = cosine_similarity(embeddings)

    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)

    similarities = sim_matrix[upper_triangle_indices]

    return float(similarities.mean())


# ==========================================================
# 3. Main pipeline
# ==========================================================

def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        local_files_only=False
    )

    print(f"Opening database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    try:
        # ----------------------------------------------------------
        # Load posts and comments
        # ----------------------------------------------------------

        print("Loading posts table...")
        posts = pd.read_sql("SELECT * FROM posts;", conn)

        print("Loading comments table...")
        comments = pd.read_sql("SELECT * FROM comments;", conn)

        print("Posts loaded:", len(posts))
        print("Comments loaded:", len(comments))

        # ----------------------------------------------------------
        # Check required columns
        # ----------------------------------------------------------

        required_post_cols = {"id", "author", "body"}
        required_comment_cols = {
            "id",
            "post_id",
            "parent_comment_id",
            "author",
            "body",
            "depth"
        }

        missing_post_cols = required_post_cols - set(posts.columns)
        missing_comment_cols = required_comment_cols - set(comments.columns)

        if missing_post_cols:
            raise KeyError(
                f"Missing required columns in posts table: {missing_post_cols}"
            )

        if missing_comment_cols:
            raise KeyError(
                f"Missing required columns in comments table: {missing_comment_cols}"
            )

        # ----------------------------------------------------------
        # Prepare posts
        # ----------------------------------------------------------

        posts_internal = posts.copy()

        posts_internal["text_type"] = "post"
        posts_internal["text_id"] = posts_internal["id"]
        posts_internal["text_body"] = posts_internal["body"]

        # Some posts tables do not contain post_id.
        # In that case, the post id itself is used as post_id.
        if "post_id" not in posts_internal.columns:
            posts_internal["post_id"] = posts_internal["id"]

        posts_internal["parent_comment_id"] = np.nan
        posts_internal["depth"] = np.nan

        # ----------------------------------------------------------
        # Prepare comments
        # ----------------------------------------------------------

        comments_internal = comments.copy()

        comments_internal["text_type"] = "comment"
        comments_internal["text_id"] = comments_internal["id"]
        comments_internal["text_body"] = comments_internal["body"]

        # ----------------------------------------------------------
        # Select common columns
        # ----------------------------------------------------------

        common_columns = [
            "text_type",
            "text_id",
            "post_id",
            "parent_comment_id",
            "author",
            "text_body",
            "depth"
        ]

        posts_internal = posts_internal[common_columns].copy()
        comments_internal = comments_internal[common_columns].copy()

        # ----------------------------------------------------------
        # Merge posts and comments into one dataframe
        # ----------------------------------------------------------

        texts_internal = pd.concat(
            [posts_internal, comments_internal],
            ignore_index=True
        )

        texts_internal["text_body"] = (
            texts_internal["text_body"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        texts_internal = texts_internal[
            texts_internal["text_body"].str.strip() != ""
        ].copy()

        print("Total non-empty texts to analyze:", len(texts_internal))

        # ----------------------------------------------------------
        # Compute internal cosine similarity
        # ----------------------------------------------------------

        print("Computing internal cosine similarity for posts and comments...")

        tqdm.pandas()

        texts_internal["internal_similarity"] = texts_internal["text_body"].progress_apply(
            lambda x: compute_internal_similarity(x, embedding_model)
        )

        # ----------------------------------------------------------
        # Save result into SQL table
        # ----------------------------------------------------------

        print(f"Saving SQL table: {OUTPUT_TABLE}")

        texts_internal.to_sql(
            OUTPUT_TABLE,
            conn,
            if_exists="replace",
            index=False
        )

        print(f"SQL table '{OUTPUT_TABLE}' saved successfully.")

        # ----------------------------------------------------------
        # Check saved table
        # ----------------------------------------------------------

        print("Preview of saved table:")

        check_internal = pd.read_sql(
            f"""
            SELECT 
                text_type,
                text_id,
                post_id,
                parent_comment_id,
                author,
                depth,
                internal_similarity,
                text_body
            FROM {OUTPUT_TABLE}
            LIMIT 20;
            """,
            conn
        )

        print(check_internal)

        # ----------------------------------------------------------
        # Basic summary
        # ----------------------------------------------------------

        print("\nSummary by text_type:")

        summary = pd.read_sql(
            f"""
            SELECT
                text_type,
                COUNT(*) AS total_texts,
                COUNT(internal_similarity) AS texts_with_internal_similarity,
                AVG(internal_similarity) AS avg_internal_similarity,
                MIN(internal_similarity) AS min_internal_similarity,
                MAX(internal_similarity) AS max_internal_similarity
            FROM {OUTPUT_TABLE}
            GROUP BY text_type;
            """,
            conn
        )

        print(summary)

    finally:
        conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    main()