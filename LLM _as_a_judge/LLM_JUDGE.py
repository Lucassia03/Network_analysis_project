# LLM_judge_extraction.py

import sqlite3
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DB_PATH = "/home/3189236/NLP_project/moltbook.db"
MODEL_ID = "allenai/Olmo-3-7B-Instruct"


def ensure_judgment_columns(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("PRAGMA table_info(comments)")
    existing_columns = [row[1] for row in cur.fetchall()]

    if "llm_judgment" not in existing_columns:
        cur.execute("ALTER TABLE comments ADD COLUMN llm_judgment TEXT")

    if "llm_judgment_explanation" not in existing_columns:
        cur.execute("ALTER TABLE comments ADD COLUMN llm_judgment_explanation TEXT")

    con.commit()
    con.close()


def generate_with_olmo(model, tokenizer, prompt, max_new_tokens=250):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Natural Language Inference judge. "
                "Your task is to classify the relation between a comment and a context."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    device = next(model.parameters()).device

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def judge_comment_against_context(model, tokenizer, comment_text: str, context_text: str):
    prompt = f"""
You are an expert Natural Language Inference judge.

Compare the COMMENT against the CONTEXT.

Choose exactly one label:

Entailment:
The context clearly supports, implies, or confirms the comment.

Contradiction:
The context clearly conflicts with, denies, or is incompatible with the comment.

Neutral:
The context is related but does not provide enough information to prove or disprove the comment.

Rules:
- Use only the provided context.
- Do not use outside knowledge.
- If the context is insufficient, choose Neutral.
- Output only valid JSON.
- The label must be exactly one of: Entailment, Contradiction, Neutral.

CONTEXT:
{context_text}

COMMENT:
{comment_text}

Return JSON exactly in this format:
{{
  "label": "Entailment",
  "explanation": "Brief explanation."
}}
"""

    raw_output = generate_with_olmo(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=250,
    )

    try:
        parsed = json.loads(raw_output)
        label = parsed.get("label", "Neutral")
        explanation = parsed.get("explanation", "")
    except json.JSONDecodeError:
        label = "Neutral"
        explanation = f"Could not parse model output as JSON. Raw output: {raw_output}"

    valid_labels = {"Entailment", "Contradiction", "Neutral"}

    if label not in valid_labels:
        label = "Neutral"

    if not explanation:
        explanation = raw_output

    return label, explanation


def judge_all_comments(db_path: str, model, tokenizer, limit: int | None = 10):
    ensure_judgment_columns(db_path)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    query = """
        SELECT id, body, weighted_context
        FROM comments
        WHERE weighted_context IS NOT NULL
          AND TRIM(weighted_context) != ''
          AND body IS NOT NULL
          AND TRIM(body) != ''
          AND llm_judgment IS NULL
        ORDER BY id
    """

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    rows = cur.execute(query).fetchall()

    print(f"Judging {len(rows)} comments...", flush=True)

    for idx, row in enumerate(rows, start=1):
        comment_id = row["id"]
        comment_text = row["body"]
        context_text = row["weighted_context"]

        print(f"Judging comment {idx}/{len(rows)} with id={comment_id}...", flush=True)

        label, explanation = judge_comment_against_context(
            model=model,
            tokenizer=tokenizer,
            comment_text=comment_text,
            context_text=context_text,
        )

        cur.execute(
            """
            UPDATE comments
            SET llm_judgment = ?,
                llm_judgment_explanation = ?
            WHERE id = ?
            """,
            (label, explanation, comment_id),
        )

        con.commit()

        print(f"Saved judgment for id={comment_id}: {label}", flush=True)

    con.close()

    print("Finished judging comments.", flush=True)


def main():
    db_path = Path(DB_PATH)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print("Loading OLMo 3 7B Instruct...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        print("CUDA GPU detected. Loading model on GPU.", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
    else:
        print("No CUDA GPU detected. Loading model on CPU. This may be slow.", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(torch.device("cpu"))

    model.eval()

    print(f"Model loaded on: {next(model.parameters()).device}", flush=True)

    # Test first with 10 comments.
    # After confirming it works, change limit=10 to limit=None.
    judge_all_comments(
        db_path=str(db_path),
        model=model,
        tokenizer=tokenizer,
        limit=10,
    )


if __name__ == "__main__":
    main()