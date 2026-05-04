import sqlite3
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ==========================================================
# 1. Load models
# ==========================================================
# Sentence embedding model for cosine similarity
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=False
)

# NLI model for contradiction / neutral / entailment classification
nli_model_name = "facebook/bart-large-mnli"

tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model.to(device)
nli_model.eval()

id2label = nli_model.config.id2label
print("NLI labels:", id2label)

# ==========================================================
# 2. Load database
# ==========================================================

db_path = "moltbook (3).db"   # Change this if your database has another name
conn = sqlite3.connect(db_path)

posts = pd.read_sql("SELECT * FROM posts;", conn)
comments = pd.read_sql("SELECT * FROM comments;", conn)

# ==========================================================
# 3. Clean and prepare text columns
# ==========================================================
# Clean post text
posts["body"] = posts["body"].fillna("").astype(str).str.lower()

# Clean comment text
comments["body"] = comments["body"].fillna("").astype(str).str.lower()

# Make sure parent_comment_id is numeric.
# Missing values will become NaN.
comments["parent_comment_id"] = pd.to_numeric(
    comments["parent_comment_id"],
    errors="coerce"
)

# ==========================================================
# 4. Build pair type 1: post -> direct comment
# ==========================================================
# Direct comments are comments whose parent_comment_id is missing.
# In this case, the parent is the original post.

post_to_comment = comments[
    comments["parent_comment_id"].isna()
].merge(
    posts,
    left_on="post_id",
    right_on="id",
    suffixes=("_child", "_post")
)

post_to_comment_pairs = pd.DataFrame({
    "relation_type": "post_to_comment",
    "post_id": post_to_comment["post_id"],
    "parent_id": post_to_comment["id_post"],
    "child_id": post_to_comment["id_child"],
    "parent_author": post_to_comment["author_post"],
    "child_author": post_to_comment["author_child"],
    "parent_text": post_to_comment["body_post"],
    "child_text": post_to_comment["body_child"],
    "child_depth": post_to_comment["depth"]
})

# ==========================================================
# 5. Build pair type 2: parent comment -> child comment
# ==========================================================
# Replies to comments have a non-null parent_comment_id.
# We perform a self-join on the comments table:
# child.parent_comment_id = parent.id

comment_to_comment = comments[
    comments["parent_comment_id"].notna()
].merge(
    comments,
    left_on="parent_comment_id",
    right_on="id",
    suffixes=("_child", "_parent")
)

comment_to_comment_pairs = pd.DataFrame({
    "relation_type": "comment_to_comment",
    "post_id": comment_to_comment["post_id_child"],
    "parent_id": comment_to_comment["id_parent"],
    "child_id": comment_to_comment["id_child"],
    "parent_author": comment_to_comment["author_parent"],
    "child_author": comment_to_comment["author_child"],
    "parent_text": comment_to_comment["body_parent"],
    "child_text": comment_to_comment["body_child"],
    "child_depth": comment_to_comment["depth_child"]
})

# ==========================================================
# 6. Merge all parent-child pairs
# ==========================================================

df_pairs = pd.concat(
    [post_to_comment_pairs, comment_to_comment_pairs],
    ignore_index=True
)

# Remove pairs with empty parent or child text
df_pairs = df_pairs[
    (df_pairs["parent_text"].str.strip() != "") &
    (df_pairs["child_text"].str.strip() != "")
].copy()

print("Number of parent-child pairs:")
print(df_pairs["relation_type"].value_counts())

# ==========================================================
# 7. Compute cosine similarity for all pairs
# ==========================================================

print("Encoding parent texts...")
parent_embeddings = embedding_model.encode(
    df_pairs["parent_text"].tolist(),
    show_progress_bar=True
)

print("Encoding child texts...")
child_embeddings = embedding_model.encode(
    df_pairs["child_text"].tolist(),
    show_progress_bar=True
)

df_pairs["similarity"] = [
    cosine_similarity([parent_emb], [child_emb])[0][0]
    for parent_emb, child_emb in zip(parent_embeddings, child_embeddings)
]

# ==========================================================
# 8. Run NLI on all parent-child pairs
# ==========================================================
# In Natural Language Inference:
# parent_text is the premise
# child_text is the hypothesis

contradiction_scores = []
neutral_scores = []
entailment_scores = []
predictions = []

batch_size = 16

print("Running NLI on all parent-child pairs...")

for i in tqdm(range(0, len(df_pairs), batch_size)):
    batch = df_pairs.iloc[i:i + batch_size]

    premises = batch["parent_text"].tolist()
    hypotheses = batch["child_text"].tolist()

    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = nli_model(**encoded)
        logits = outputs.logits

    logits = logits.cpu()

    for row in logits:
        scores = {
            id2label[j].lower(): row[j].item()
            for j in range(len(row))
        }

        contradiction_score = scores["contradiction"]
        neutral_score = scores["neutral"]
        entailment_score = scores["entailment"]

        contradiction_scores.append(contradiction_score)
        neutral_scores.append(neutral_score)
        entailment_scores.append(entailment_score)

        prediction = max(scores, key=scores.get)
        predictions.append(prediction)

df_pairs["contradiction_score"] = contradiction_scores
df_pairs["neutral_score"] = neutral_scores
df_pairs["entailment_score"] = entailment_scores
df_pairs["prediction"] = predictions

# ==========================================================
# 9. Save final output into SQL table
# ==========================================================

table_name = "parent_child_similarity"

df_pairs.to_sql(
    table_name,
    conn,
    if_exists="replace",
    index=False
)

print(f"SQL table '{table_name}' saved successfully.")

# ==========================================================
# 10. Check saved table
# ==========================================================

check = pd.read_sql(
    f"""
    SELECT 
        relation_type,
        post_id,
        parent_id,
        child_id,
        parent_author,
        child_author,
        similarity,
        contradiction_score,
        neutral_score,
        entailment_score,
        prediction
    FROM {table_name}
    LIMIT 20;
    """,
    conn
)

print(check)

conn.close()