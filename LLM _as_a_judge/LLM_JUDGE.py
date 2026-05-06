# LLM_judge_extraction.py

import sqlite3
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------
# Paths and model
# ------------------------------------------------------------

DB_PATH = "/home/3189236/NLP_project/moltbook.db"
MODEL_ID = "allenai/Olmo-3-7B-Instruct"


# ------------------------------------------------------------
# Database utilities
# ------------------------------------------------------------

def ensure_judgment_columns(db_path: str):
    """
    Adds the LLM judgment columns to the comments table if they do not exist.
    """

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("PRAGMA table_info(comments)")
    existing_columns = [row[1] for row in cur.fetchall()]

    if "LLM_judgment" not in existing_columns:
        cur.execute("ALTER TABLE comments ADD COLUMN LLM_judgment TEXT")

    if "LLM_judgment_explanation" not in existing_columns:
        cur.execute("ALTER TABLE comments ADD COLUMN LLM_judgment_explanation TEXT")

    con.commit()
    con.close()


def preview_existing_columns(db_path: str):
    """
    Prints the columns in the comments table.
    Useful for checking that LLM_judgment exists.
    """

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("PRAGMA table_info(comments)")
    columns = [row[1] for row in cur.fetchall()]

    con.close()

    print("Columns in comments table:", flush=True)
    for col in columns:
        print(f"- {col}", flush=True)


# ------------------------------------------------------------
# Model generation
# ------------------------------------------------------------

def generate_with_olmo(model, tokenizer, prompt: str, max_new_tokens: int = 250):
    """
    Sends a prompt to OLMo and returns generated text.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Natural Language Inference judge. "
                "You classify whether a comment is Entailment, Contradiction, or Neutral "
                "with respect to a given context."
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
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ------------------------------------------------------------
# Output parsing
# ------------------------------------------------------------

def extract_json_object(text: str):
    """
    Tries to extract a JSON object from the model output.
    """

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def normalize_label(label: str):
    """
    Normalizes model labels to exactly:
    Entailment, Contradiction, or Neutral.
    """

    if not label:
        return "Neutral"

    cleaned = label.strip().lower()

    if "entail" in cleaned:
        return "Entailment"

    if "contradict" in cleaned:
        return "Contradiction"

    if "neutral" in cleaned:
        return "Neutral"

    return "Neutral"


# ------------------------------------------------------------
# LLM judge
# ------------------------------------------------------------

def judge_comment_against_context(
    model,
    tokenizer,
    comment_text: str,
    context_text: str,
):
    """
    Uses OLMo as a judge to classify the relation between context and comment.
    """

    prompt = f"""
You are an expert Natural Language Inference judge.

Your task is to compare the COMMENT against the CONTEXT.

Classify the relationship as exactly one of these labels:

Entailment:
The context clearly supports, implies, or confirms the comment.

Contradiction:
The context clearly conflicts with, denies, or is incompatible with the comment.

Neutral:
The context is related but does not provide enough information to prove or disprove the comment.

Important rules:
- Use only the provided context.
- Do not use outside knowledge.
- If the context is insufficient, choose Neutral.
- Do not invent facts.
- Output only valid JSON.
- The label must be exactly one of: Entailment, Contradiction, Neutral.

CONTEXT:
{context_text}

COMMENT:
{comment_text}

Return JSON exactly in this format:
{{
  "label": "Entailment",
  "explanation": "Brief explanation of the decision."
}}
"""

    raw_output = generate_with_olmo(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=250,
    )

    parsed = extract_json_object(raw_output)

    if parsed is None:
        label = "Neutral"
        explanation = f"Could not parse model output as JSON. Raw output: {raw_output}"
    else:
        label = normalize_label(parsed.get("label", "Neutral"))
        explanation = parsed.get("explanation", "").strip()

        if not explanation:
            explanation = f"Model returned label {label} without explanation."

    return label, explanation


# ------------------------------------------------------------
# Main judging loop
# ------------------------------------------------------------

def judge_all_comments(db_path: str, model, tokenizer, limit: int | None = 10):
    """
    Reads comments and weighted_context from the database,
    judges each comment,
    and saves the result in comments.LLM_judgment.
    """

    ensure_judgment_columns(db_path)
    preview_existing_columns(db_path)

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
          AND LLM_judgment IS NULL
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

        print(
            f"Judging comment {idx}/{len(rows)} with id={comment_id}...",
            flush=True,
        )

        try:
            label, explanation = judge_comment_against_context(
                model=model,
                tokenizer=tokenizer,
                comment_text=comment_text,
                context_text=context_text,
            )

            cur.execute(
                """
                UPDATE comments
                SET LLM_judgment = ?,
                    LLM_judgment_explanation = ?
                WHERE id = ?
                """,
                (label, explanation, comment_id),
            )

            con.commit()

            print(
                f"Saved judgment for id={comment_id}: {label}",
                flush=True,
            )

        except Exception as e:
            error_message = f"ERROR while judging comment id={comment_id}: {repr(e)}"

            cur.execute(
                """
                UPDATE comments
                SET LLM_judgment = ?,
                    LLM_judgment_explanation = ?
                WHERE id = ?
                """,
                ("Neutral", error_message, comment_id),
            )

            con.commit()

            print(error_message, flush=True)

    con.close()

    print("Finished judging comments.", flush=True)


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

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

    # First test with 10 comments.
    # After confirming it works, change limit=10 to limit=None.
    judge_all_comments(
        db_path=str(db_path),
        model=model,
        tokenizer=tokenizer,
        limit=10,
    )


if __name__ == "__main__":
    main()
