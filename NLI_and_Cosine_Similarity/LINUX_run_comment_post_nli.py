import sqlite3
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ==========================================================
# 1. Settings
# ==========================================================

db_path = "moltbook_final_NLI_sim.db"   # nome del tuo database
table_name = "comment_post_nli"         # tabella di output

nli_model_name = "facebook/bart-large-mnli"
batch_size = 16


# ==========================================================
# 2. Load NLI model
# ==========================================================

print("Loading NLI model...")

tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model.to(device)
nli_model.eval()

id2label = nli_model.config.id2label
print("NLI labels:", id2label)
print("Using device:", device)


# ==========================================================
# 3. Load database
# ==========================================================

print("Loading database...")

conn = sqlite3.connect(db_path)

posts = pd.read_sql("SELECT * FROM posts;", conn)
comments = pd.read_sql("SELECT * FROM comments;", conn)


# ==========================================================
# 4. Clean and prepare text columns
# ==========================================================

posts["title"] = posts["title"].fillna("").astype(str)
posts["body"] = posts["body"].fillna("").astype(str)

comments["body"] = comments["body"].fillna("").astype(str)

# Creo un testo completo del post: titolo + body
posts["post_text"] = (
    posts["title"].str.strip() + "\n" + posts["body"].str.strip()
).str.strip().str.lower()

comments["comment_text"] = comments["body"].str.strip().str.lower()

# Pulizia id parent, anche se qui non serve per costruire le coppie
comments["parent_comment_id"] = pd.to_numeric(
    comments["parent_comment_id"],
    errors="coerce"
)


# ==========================================================
# 5. Build pairs: post -> every comment of that post
# ==========================================================
# Qui ogni commento viene confrontato con il post originale.
# Non importa se il commento è diretto o una risposta annidata:
# se ha lo stesso post_id, viene confrontato con quel post.

df_pairs = comments.merge(
    posts,
    left_on="post_id",
    right_on="id",
    suffixes=("_comment", "_post")
)

df_pairs = pd.DataFrame({
    "relation_type": "comment_to_original_post",
    "post_id": df_pairs["post_id"],
    "post_author": df_pairs["author_post"],
    "comment_id": df_pairs["id_comment"],
    "comment_author": df_pairs["author_comment"],
    "parent_comment_id": df_pairs["parent_comment_id"],
    "comment_depth": df_pairs["depth"],
    "post_text": df_pairs["post_text"],
    "comment_text": df_pairs["comment_text"]
})

# Rimuove righe con post o commento vuoto
df_pairs = df_pairs[
    (df_pairs["post_text"].str.strip() != "") &
    (df_pairs["comment_text"].str.strip() != "")
].copy()

print("Number of comment-post pairs:")
print(len(df_pairs))


# ==========================================================
# 6. Run NLI on all comment-post pairs
# ==========================================================
# In Natural Language Inference:
# post_text is the premise
# comment_text is the hypothesis

contradiction_scores = []
neutral_scores = []
entailment_scores = []
predictions = []

print("Running NLI on all comment-post pairs...")

for i in tqdm(range(0, len(df_pairs), batch_size)):
    batch = df_pairs.iloc[i:i + batch_size]

    premises = batch["post_text"].tolist()
    hypotheses = batch["comment_text"].tolist()

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
# 7. Save final output into SQL table
# ==========================================================

df_pairs.to_sql(
    table_name,
    conn,
    if_exists="replace",
    index=False
)

print(f"SQL table '{table_name}' saved successfully in database '{db_path}'.")


# ==========================================================
# 8. Check saved table
# ==========================================================

check = pd.read_sql(
    f"""
    SELECT 
        relation_type,
        post_id,
        comment_id,
        post_author,
        comment_author,
        parent_comment_id,
        comment_depth,
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

print("Done.")