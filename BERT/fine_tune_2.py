import os
import json
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# Paths

file_path = "FILE_FINALE_TRAINING.xlsx"
save_dir = "/mnt/beegfsstudents/home/3194097/NLI_project/Storing_training"

os.makedirs(save_dir, exist_ok=True)

# 1) Load data
df = pd.read_excel(file_path, engine="openpyxl", sheet_name="output")
df = df[["text", "model"]].dropna().copy()
df["text"] = df["text"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()
df = df[df["text"] != ""]

# 2) Encode labels
label_list = sorted(df["model"].unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["model"].map(label2id)

print("Labels:", label2id)
print(df["model"].value_counts())

# 3) Train/validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

# 4) Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

train_ds = train_ds.map(preprocess_function, batched=True)
val_ds = val_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5) Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 6) Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted")
    }

# 7) Training arguments
training_args = TrainingArguments(
    output_dir=save_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none"
)

# 8) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 9) Train
trainer.train()

# 10) Evaluate
results = trainer.evaluate()
print("Evaluation results:", results)

# 11) Save final best model and tokenizer
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

with open(os.path.join(save_dir, "label_mappings.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()}
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Model saved in: {save_dir}")

# 12) Reload and test
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForSequenceClassification.from_pretrained(save_dir)
model.eval()

def predict_model(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()

    return {
        "predicted_model": model.config.id2label[pred_id],
        "confidence": confidence
    }

example = "Write a short paragraph about renewable energy and its benefits."
print(predict_model(example))
