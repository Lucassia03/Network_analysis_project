import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
DB_PATH = "/home/3189236/NLP_project/moltbook_final_NLI_sim.db"

LIMIT = None
OVERWRITE = True
PROCESS_ORDER_COLUMN = "id"
RESET_OUTPUT_COLUMNS = True

MAX_POST_TITLE_CHARS = 300
MAX_POST_BODY_CHARS = 1200
MAX_COMMENT_CHARS = 900
MAX_PARENT_CHARS = 900
MAX_ANCESTORS = 5
MAX_ANCESTOR_CHARS = 700
MAX_PREVIOUS_SIBLINGS = 5
MAX_SIBLING_CHARS = 600
MAX_EARLIER_COMMENTS = 8
MAX_EARLIER_COMMENT_CHARS = 500
MAX_NEW_TOKENS = 320

OUTPUT_COLUMNS = {
    "LLM_External_Reference_Detected": "TEXT",
    "LLM_External_Reference_Type": "TEXT",
    "LLM_External_Reference_Target": "TEXT",
    "LLM_External_Reference_Explanation": "TEXT",
    "LLM_External_Reference_Confidence": "REAL",
}

VALID_DETECTED = {"Yes", "No", "Unclear"}
VALID_TYPES = {
    "Previous discussion",
    "External agent/entity",
    "Both",
    "None",
    "Unclear",
}


def trunc(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def is_empty_or_deleted(text: Any) -> bool:
    if text is None:
        return True
    cleaned = str(text).strip().lower()
    return cleaned in {"", "[deleted]", "[removed]", "deleted", "removed"}


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def get_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [row[1] for row in rows]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def ensure_output_columns(conn: sqlite3.Connection) -> None:
    existing = set(get_columns(conn, "comments"))
    for column, col_type in OUTPUT_COLUMNS.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE comments ADD COLUMN {quote_ident(column)} {col_type}"
            )
            print(f"Added column: {column}", flush=True)
    conn.commit()


def reset_output_columns(conn: sqlite3.Connection) -> None:
    existing = set(get_columns(conn, "comments"))
    dropped = []

    for column in OUTPUT_COLUMNS:
        if column in existing:
            conn.execute(f"ALTER TABLE comments DROP COLUMN {quote_ident(column)}")
            dropped.append(column)

    conn.commit()

    for column in dropped:
        print(f"Dropped column: {column}", flush=True)


def choose_order_column(columns: List[str]) -> Optional[str]:
    for candidate in [
        "comment_order",
        "timestamp_order",
        "created_utc",
        "created",
        "timestamp",
        "id",
    ]:
        if candidate in columns:
            return candidate
    return None


def get_row_by_id(
    conn: sqlite3.Connection,
    table: str,
    row_id: Any,
) -> Optional[sqlite3.Row]:
    if row_id is None:
        return None
    try:
        return conn.execute(
            f"SELECT * FROM {quote_ident(table)} WHERE id = ?",
            (row_id,),
        ).fetchone()
    except sqlite3.Error:
        return None


def get_post_context(
    conn: sqlite3.Connection,
    post_id: Any,
) -> Tuple[str, str]:
    if not table_exists(conn, "posts") or post_id is None:
        return "", ""

    post_columns = set(get_columns(conn, "posts"))
    if "id" not in post_columns:
        return "", ""

    row = get_row_by_id(conn, "posts", post_id)
    if row is None:
        return "", ""

    title = row["title"] if "title" in post_columns else ""
    body = row["body"] if "body" in post_columns else ""
    return trunc(title, MAX_POST_TITLE_CHARS), trunc(body, MAX_POST_BODY_CHARS)


def format_comment(row: Optional[sqlite3.Row], columns: List[str], max_chars: int) -> str:
    if row is None:
        return "Not available."

    pieces = []
    if "id" in columns:
        pieces.append(f"id={row['id']}")
    if "author" in columns and row["author"]:
        pieces.append(f"author={row['author']}")

    body = row["body"] if "body" in columns else ""
    body_text = trunc(body, max_chars)
    if is_empty_or_deleted(body_text):
        body_text = "[empty/deleted]"

    prefix = " | ".join(pieces)
    if prefix:
        return f"{prefix}: {body_text}"
    return body_text


def get_parent_comment(
    conn: sqlite3.Connection,
    comment: sqlite3.Row,
    columns: List[str],
) -> Optional[sqlite3.Row]:
    if "parent_comment_id" not in columns:
        return None
    parent_id = comment["parent_comment_id"]
    if parent_id is None or parent_id == "":
        return None
    return get_row_by_id(conn, "comments", parent_id)


def get_ancestor_chain(
    conn: sqlite3.Connection,
    comment: sqlite3.Row,
    columns: List[str],
) -> List[sqlite3.Row]:
    if "parent_comment_id" not in columns:
        return []

    ancestors = []
    seen = {comment["id"]} if "id" in columns else set()
    current = comment

    for _ in range(MAX_ANCESTORS):
        parent_id = current["parent_comment_id"]
        if parent_id is None or parent_id == "" or parent_id in seen:
            break
        parent = get_row_by_id(conn, "comments", parent_id)
        if parent is None:
            break
        ancestors.append(parent)
        seen.add(parent_id)
        current = parent

    return ancestors


def get_previous_siblings(
    conn: sqlite3.Connection,
    comment: sqlite3.Row,
    columns: List[str],
    order_col: Optional[str],
) -> List[sqlite3.Row]:
    if (
        "parent_comment_id" not in columns
        or "post_id" not in columns
        or "id" not in columns
        or order_col is None
    ):
        return []

    parent_id = comment["parent_comment_id"]
    post_id = comment["post_id"]
    order_value = comment[order_col]

    where_parent = (
        "parent_comment_id IS NULL"
        if parent_id is None
        else "parent_comment_id = ?"
    )
    params: List[Any] = [post_id]
    if parent_id is not None:
        params.append(parent_id)
    params.append(order_value)
    params.append(comment["id"])
    params.append(MAX_PREVIOUS_SIBLINGS)

    sql = f"""
        SELECT *
        FROM comments
        WHERE post_id = ?
          AND {where_parent}
          AND (
                {quote_ident(order_col)} < ?
                OR ({quote_ident(order_col)} = ? AND id < ?)
          )
        ORDER BY {quote_ident(order_col)} DESC, id DESC
        LIMIT ?
    """
    try:
        rows = conn.execute(sql, tuple(params)).fetchall()
        return list(reversed(rows))
    except sqlite3.Error:
        return []


def get_earlier_same_post_comments(
    conn: sqlite3.Connection,
    comment: sqlite3.Row,
    columns: List[str],
    order_col: Optional[str],
) -> List[sqlite3.Row]:
    if "post_id" not in columns or "id" not in columns or order_col is None:
        return []

    post_id = comment["post_id"]
    order_value = comment[order_col]

    sql = f"""
        SELECT *
        FROM comments
        WHERE post_id = ?
          AND (
                {quote_ident(order_col)} < ?
                OR ({quote_ident(order_col)} = ? AND id < ?)
          )
        ORDER BY {quote_ident(order_col)} DESC, id DESC
        LIMIT ?
    """
    try:
        rows = conn.execute(
            sql,
            (post_id, order_value, order_value, comment["id"], MAX_EARLIER_COMMENTS),
        ).fetchall()
        return list(reversed(rows))
    except sqlite3.Error:
        return []


def build_prompt(
    conn: sqlite3.Connection,
    comment: sqlite3.Row,
    columns: List[str],
    order_col: Optional[str],
) -> str:
    post_title, post_body = get_post_context(
        conn,
        comment["post_id"] if "post_id" in columns else None,
    )

    parent = get_parent_comment(conn, comment, columns)
    ancestors = get_ancestor_chain(conn, comment, columns)
    previous_siblings = get_previous_siblings(conn, comment, columns, order_col)
    earlier_comments = get_earlier_same_post_comments(conn, comment, columns, order_col)

    target_body = comment["body"] if "body" in columns else ""
    if is_empty_or_deleted(target_body):
        target_body = "[empty/deleted]"

    ancestor_text = "\n".join(
        f"- {format_comment(row, columns, MAX_ANCESTOR_CHARS)}" for row in reversed(ancestors)
    )
    sibling_text = "\n".join(
        f"- {format_comment(row, columns, MAX_SIBLING_CHARS)}" for row in previous_siblings
    )
    earlier_text = "\n".join(
        f"- {format_comment(row, columns, MAX_EARLIER_COMMENT_CHARS)}"
        for row in earlier_comments
    )

    return f"""
You are a careful annotation model. Judge only whether the TARGET COMMENT refers to something external to the current post context.

Mark "Yes" if the target comment appears to refer to:
- a previous discussion, previous thread, previous post, earlier debate, older comment, or conversation outside the current post; or
- an agent, person, user, organization, entity, system, model, bot, operator, author, or actor that is not introduced or grounded in the current post title, post body, parent comment, ancestor comments, previous sibling comments, or earlier comments from the same post.

Mark "No" if the reference is clearly grounded in the current post title/body, direct parent comment, ancestor comments, previous sibling comments, earlier comments from the same post, or the comment only replies locally.

Mark "Unclear" if vague references such as "he", "she", "they", "that", "this", "the agent", "the guy", "the model", "the bot", "the operator", or "we discussed this" cannot be resolved from the provided local context.

Do not classify sentiment, toxicity, ideology, morality, political position, correctness, or agreement.

Return valid JSON only using exactly this schema:
{{
  "external_reference_detected": "Yes",
  "external_reference_type": "Previous discussion",
  "external_reference_target": "short description",
  "explanation": "brief explanation",
  "confidence": 0.85
}}

Valid external_reference_detected values: "Yes", "No", "Unclear".
Valid external_reference_type values: "Previous discussion", "External agent/entity", "Both", "None", "Unclear".
Confidence must be a float between 0 and 1.

TARGET COMMENT:
{trunc(target_body, MAX_COMMENT_CHARS)}

CURRENT POST CONTEXT:
Title: {post_title if post_title else "Not available."}
Body: {post_body if post_body else "Not available."}

DIRECT PARENT COMMENT:
{format_comment(parent, columns, MAX_PARENT_CHARS)}

ANCESTOR CHAIN:
{ancestor_text if ancestor_text else "None available."}

PREVIOUS SIBLINGS:
{sibling_text if sibling_text else "None available."}

EARLIER SAME-POST COMMENTS:
{earlier_text if earlier_text else "None available."}
""".strip()


def load_model_and_tokenizer():
    print(f"Loading tokenizer: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        print("CUDA available. Loading model on GPU 0 with float16.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        print("CUDA not available. Loading model on CPU with float32.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None

    return None


def normalize_result(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "external_reference_detected": "Unclear",
            "external_reference_type": "Unclear",
            "external_reference_target": "",
            "explanation": "Model did not return parseable JSON.",
            "confidence": 0.0,
        }

    detected = str(raw.get("external_reference_detected", "Unclear")).strip()
    ref_type = str(raw.get("external_reference_type", "Unclear")).strip()

    if detected not in VALID_DETECTED:
        detected = "Unclear"
    if ref_type not in VALID_TYPES:
        ref_type = "Unclear"

    if detected == "No" and ref_type not in {"None", "Unclear"}:
        ref_type = "None"
    if detected == "Yes" and ref_type == "None":
        ref_type = "Unclear"

    target = raw.get("external_reference_target", "")
    if target is None:
        target = ""
    target = trunc(target, 300)

    explanation = raw.get("explanation", "")
    if explanation is None:
        explanation = ""
    explanation = trunc(explanation, 1000)

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "external_reference_detected": detected,
        "external_reference_type": ref_type,
        "external_reference_target": target,
        "explanation": explanation,
        "confidence": confidence,
    }


def judge_comment(tokenizer, model, prompt: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "You are a precise JSON-only annotation model.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    encoded = encoded.to(model_device)

    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    parsed = extract_first_json_object(decoded)
    return normalize_result(parsed)


def fetch_comments_to_process(
    conn: sqlite3.Connection,
    columns: List[str],
    process_order_col: Optional[str],
) -> List[sqlite3.Row]:
    where_parts = []
    if not OVERWRITE and "LLM_External_Reference_Detected" in columns:
        if "LLM_External_Reference_Explanation" in columns:
            where_parts.append(
                "("
                "LLM_External_Reference_Detected IS NULL "
                "OR LLM_External_Reference_Explanation LIKE 'Processing error:%'"
                ")"
            )
        else:
            where_parts.append("LLM_External_Reference_Detected IS NULL")

    where_sql = ""
    if where_parts:
        where_sql = "WHERE " + " AND ".join(where_parts)

    order_sql = ""
    if process_order_col is not None:
        order_sql = f"ORDER BY {quote_ident(process_order_col)} ASC"
    elif "id" in columns:
        order_sql = "ORDER BY id ASC"

    limit_sql = ""
    params: Tuple[Any, ...] = ()
    if LIMIT is not None:
        limit_sql = "LIMIT ?"
        params = (LIMIT,)

    sql = f"SELECT * FROM comments {where_sql} {order_sql} {limit_sql}"
    return conn.execute(sql, params).fetchall()


def update_comment_result(
    conn: sqlite3.Connection,
    comment_id: Any,
    result: Dict[str, Any],
) -> None:
    conn.execute(
        """
        UPDATE comments
        SET
            LLM_External_Reference_Detected = ?,
            LLM_External_Reference_Type = ?,
            LLM_External_Reference_Target = ?,
            LLM_External_Reference_Explanation = ?,
            LLM_External_Reference_Confidence = ?
        WHERE id = ?
        """,
        (
            result["external_reference_detected"],
            result["external_reference_type"],
            result["external_reference_target"],
            result["explanation"],
            result["confidence"],
            comment_id,
        ),
    )
    conn.commit()


def main() -> None:
    print(f"Opening database: {DB_PATH}", flush=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        if not table_exists(conn, "comments"):
            raise RuntimeError("Database does not contain a comments table.")

        if RESET_OUTPUT_COLUMNS:
            reset_output_columns(conn)
        ensure_output_columns(conn)
        columns = get_columns(conn, "comments")

        required = {"id", "body"}
        missing_required = required - set(columns)
        if missing_required:
            raise RuntimeError(
                f"comments table is missing required columns: {sorted(missing_required)}"
            )

        context_order_col = choose_order_column(columns)
        if PROCESS_ORDER_COLUMN in columns:
            process_order_col = PROCESS_ORDER_COLUMN
        else:
            process_order_col = context_order_col

        print(f"Using context order column: {context_order_col}", flush=True)
        print(f"Using processing order column: {process_order_col}", flush=True)
        print(
            f"LIMIT={LIMIT}, OVERWRITE={OVERWRITE}, "
            f"RESET_OUTPUT_COLUMNS={RESET_OUTPUT_COLUMNS}",
            flush=True,
        )

        comments = fetch_comments_to_process(conn, columns, process_order_col)
        print(f"Comments to process: {len(comments)}", flush=True)

        if not comments:
            print("No comments to process.", flush=True)
            return

        tokenizer, model = load_model_and_tokenizer()

        for index, comment in enumerate(comments, start=1):
            comment_id = comment["id"]
            print(f"[{index}/{len(comments)}] Processing comment id={comment_id}", flush=True)

            try:
                if is_empty_or_deleted(comment["body"]):
                    result = {
                        "external_reference_detected": "Unclear",
                        "external_reference_type": "Unclear",
                        "external_reference_target": "",
                        "explanation": "Comment body is empty, deleted, or removed.",
                        "confidence": 0.0,
                    }
                else:
                    prompt = build_prompt(conn, comment, columns, context_order_col)
                    result = judge_comment(tokenizer, model, prompt)

                update_comment_result(conn, comment_id, result)
                print(
                    f"[{index}/{len(comments)}] Saved id={comment_id}: "
                    f"{result['external_reference_detected']} | "
                    f"{result['external_reference_type']} | "
                    f"confidence={result['confidence']:.2f}",
                    flush=True,
                )

            except Exception as exc:
                fallback = {
                    "external_reference_detected": "Unclear",
                    "external_reference_type": "Unclear",
                    "external_reference_target": "",
                    "explanation": f"Processing error: {type(exc).__name__}: {exc}",
                    "confidence": 0.0,
                }
                update_comment_result(conn, comment_id, fallback)
                print(
                    f"[{index}/{len(comments)}] Error on id={comment_id}: {exc}",
                    flush=True,
                )

        print("Done.", flush=True)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
