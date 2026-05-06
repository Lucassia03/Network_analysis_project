# synthesize_weighted_context.py

# LLM_Final_extraction.py

import sqlite3
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

DB_PATH = "/home/3189236/NLP_project/moltbook.db"
OUTPUT_PATH = "/home/3189236/NLP_project/unique_context_text.txt"


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_ID = "allenai/Olmo-3-7B-Instruct"


# ------------------------------------------------------------
# Load weighted context from SQLite database
# ------------------------------------------------------------

def load_weighted_contexts(db_path: str, limit: int | None = None):
    """
    Reads weighted_context from the comments table and returns cleaned text snippets.
    """

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    query = """
        SELECT
            id,
            body,
            weighted_context
        FROM comments
        WHERE weighted_context IS NOT NULL
          AND TRIM(weighted_context) != ''
        ORDER BY id
    """

    if limit:
        query += f" LIMIT {int(limit)}"

    rows = cur.execute(query).fetchall()
    con.close()

    contexts = []

    for row in rows:
        comment_id = row["id"]
        comment_body = row["body"] or ""
        raw_context = row["weighted_context"]

        try:
            parsed = json.loads(raw_context)
        except json.JSONDecodeError:
            contexts.append(
                f"Comment ID: {comment_id}\n"
                f"Target comment: {comment_body}\n"
                f"Context: {raw_context}"
            )
            continue

        evidence_parts = []

        for item in parsed:
            source = item.get("source", "unknown_source")
            weight = item.get("relevance_weight", "unknown_weight")
            excerpt = item.get("text_excerpt", "")
            summary = item.get("evidence_summary", "")

            evidence_parts.append(
                f"- Source: {source}\n"
                f"  Weight: {weight}\n"
                f"  Excerpt: {excerpt}\n"
                f"  Why relevant: {summary}"
            )

        contexts.append(
            f"Comment ID: {comment_id}\n"
            f"Target comment: {comment_body}\n"
            f"Weighted evidence:\n"
            + "\n".join(evidence_parts)
        )

    return contexts


# ------------------------------------------------------------
# Chunk text to avoid sending too much context at once
# ------------------------------------------------------------

def chunk_texts(texts, max_chars=3000):
    """
    Simple character-based chunking.

    I use 3000 characters instead of 12000 because OLMo 3 7B may be heavy
    on a local machine. You can increase this later if your computer handles it.
    """

    chunks = []
    current = []
    current_len = 0

    for text in texts:
        text_len = len(text)

        if current and current_len + text_len > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(text)
        current_len += text_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ------------------------------------------------------------
# Generate text with OLMo
# ------------------------------------------------------------

def generate_with_olmo(model, tokenizer, prompt, max_new_tokens=400):
    """
    Sends a prompt to OLMo and returns the generated text.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You synthesize database comment context into one coherent, unique text. "
                "Do not copy the input verbatim. Preserve the main ideas, relationships, "
                "tone, and important entities. Avoid bullet points unless requested."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Get the real device where the model is loaded.
    # This avoids using model.device, which can be unreliable with large models.
    device = next(model.parameters()).device

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move input tensors to the same device as the model.
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()



def save_final_text_to_db(db_path: str, final_text: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_generated_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            generated_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        INSERT INTO llm_generated_texts (model_id, generated_text)
        VALUES (?, ?)
    """, (MODEL_ID, final_text))

    con.commit()
    con.close()

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

def main():
    db_path = Path(DB_PATH)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print("Loading weighted contexts from database...", flush=True)

    # For testing, you can use:
    # contexts = load_weighted_contexts(str(db_path), limit=10)
    #
    # For the full database, use:
    contexts = load_weighted_contexts(str(db_path))

    if not contexts:
        raise ValueError("No non-empty weighted_context values found in comments table.")

    print(f"Loaded {len(contexts)} weighted contexts.", flush=True)

    print("Loading OLMo 3 7B Instruct...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print("CUDA GPU detected. Loading model on GPU.", flush=True)
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("No CUDA GPU detected. Loading model on CPU. This may be slow.", flush=True)

    print("Starting model.from_pretrained()...", flush=True)

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)

    model.eval()

    print(f"Model loaded on: {next(model.parameters()).device}", flush=True)

    chunks = chunk_texts(contexts, max_chars=3000)

    print(f"Split data into {len(chunks)} chunk(s).", flush=True)

    partial_summaries = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Generating synthesis for chunk {i}/{len(chunks)}...", flush=True)

        prompt = f"""
You are given weighted context extracted from comments in a database.

Your task:
Create a coherent synthesized text that accounts for the context.

Requirements:
- The output must be unique.
- Do not copy the input verbatim.
- Preserve the strongest recurring themes.
- Preserve important relationships, meanings, entities, and tensions.
- Give more importance to higher-weight evidence.
- Write in natural paragraph form.

DATABASE CONTEXT:
{chunk}

Write the synthesized text:
"""

        partial = generate_with_olmo(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=400,
        )

        partial_summaries.append(partial)

    if len(partial_summaries) == 1:
        final_text = partial_summaries[0]

    else:
        print("Combining chunk-level syntheses into one final text...", flush=True)

        partial_summaries_text = "\n".join(
            [
                f"--- PART {i + 1} ---\n{text}"
                for i, text in enumerate(partial_summaries)
            ]
        )

        combined_prompt = f"""
You are given several partial syntheses created from weighted comment contexts.

Your task:
Merge them into one final unique text.

Requirements:
- Remove repetition.
- Preserve the central ideas.
- Preserve important tensions and relationships.
- Do not copy the partial texts word for word.
- Write one coherent final text in natural paragraph form.

PARTIAL SYNTHESES:
{partial_summaries_text}

Final unique synthesized text:
"""

        final_text = generate_with_olmo(
            model=model,
            tokenizer=tokenizer,
            prompt=combined_prompt,
            max_new_tokens=600,
        )

    output_path = Path(OUTPUT_PATH)
    output_path.write_text(final_text, encoding="utf-8")

    print(f"\nSaved final text to: {output_path}", flush=True)
    print("\nFinal synthesized text:\n", flush=True)
    print(final_text, flush=True)
if __name__ == "__main__":
    main()