import os
import sqlite3
import pandas as pd
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# Paths
# =========================
model_dir = r"C:\Users\capod\Downloads\NLP_project\Storing_training"
db_path = r"C:\Users\capod\Downloads\NLP_project\moltbook_final_NLI_sim.db"

output_table = "comments_model_predictions"


# =========================
# Choose what to classify
# =========================
table_to_classify = "comments"
text_column = "body"

# Other possible options:
# table_to_classify = "posts"
# text_column = "body"
#
# table_to_classify = "parent_child_similarity"
# text_column = "child_text"


# =========================
# Load database
# =========================
conn = sqlite3.connect(db_path)

df = pd.read_sql_query(
    f"SELECT * FROM {table_to_classify};",
    conn
)

if text_column not in df.columns:
    raise ValueError(
        f"Column '{text_column}' not found in table '{table_to_classify}'. "
        f"Available columns: {list(df.columns)}"
    )

df[text_column] = df[text_column].astype(str).str.strip()
df = df[df[text_column] != ""].copy()

print(f"Loaded table: {table_to_classify}")
print(f"Text column: {text_column}")
print(f"Rows to classify: {len(df)}")


# =========================
# Load trained model
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Using device: {device}")


# =========================
# Predict with progress bar
# =========================
def predict_batch(texts, batch_size=32, max_length=128):
    predictions = []
    confidences = []

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(texts), batch_size),
        total=total_batches,
        desc="Classifying rows"
    ):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            pred_ids = torch.argmax(probs, dim=-1)
            batch_confidences = probs.max(dim=-1).values

        for pred_id, confidence in zip(pred_ids, batch_confidences):
            pred_id = pred_id.item()
            predictions.append(model.config.id2label[pred_id])
            confidences.append(confidence.item())

    return predictions, confidences


# =========================
# Apply classifier
# =========================
texts = df[text_column].tolist()

predicted_models, confidences = predict_batch(
    texts,
    batch_size=32,
    max_length=128
)

df["predicted_model"] = predicted_models
df["prediction_confidence"] = confidences


# =========================
# Save predictions to database
# =========================
df.to_sql(
    output_table,
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Classification completed.")
print(f"Predictions saved to table: {output_table}")

print(df[[text_column, "predicted_model", "prediction_confidence"]].head())
