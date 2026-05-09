import argparse
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


# =========================================================
# CONFIG
# =========================================================

DEFAULT_MODEL_PATH = Path("agent_classifier_pipeline.joblib")
DEFAULT_INPUT_DB = Path("moltbook_final_NLI_sim.db")
DEFAULT_OUTPUT_DB = Path("moltbook_final_NLI_sim.db")
DEFAULT_SQLITE_OUTPUT_TABLE = "agent_predictions"


# =========================================================
# JSON-safe console printing only
# =========================================================

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def print_json(payload, exit_code=0):
    print(json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2))
    sys.exit(exit_code)


# =========================================================
# Utility
# =========================================================

def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def get_existing_columns(db_path: Path, table_name: str) -> list[str]:
    conn = sqlite3.connect(db_path)

    try:
        columns = pd.read_sql(
            f'PRAGMA table_info("{table_name}");',
            conn
        )["name"].tolist()
    finally:
        conn.close()

    return columns


def read_sqlite_comments(
    db_path: Path,
    table_name: str,
    id_col: str,
    text_col: str,
    extra_cols=None,
    where_clause: str | None = None,
) -> pd.DataFrame:
    extra_cols = extra_cols or []

    existing_cols = get_existing_columns(db_path, table_name)

    if id_col not in existing_cols:
        raise KeyError(
            f"ID column '{id_col}' not found in table '{table_name}'. "
            f"Available columns: {existing_cols}"
        )

    if text_col not in existing_cols:
        raise KeyError(
            f"Text column '{text_col}' not found in table '{table_name}'. "
            f"Available columns: {existing_cols}"
        )

    valid_extra_cols = [
        col for col in extra_cols
        if col in existing_cols and col not in {id_col, text_col}
    ]

    missing_extra_cols = [
        col for col in extra_cols
        if col not in existing_cols
    ]

    if missing_extra_cols:
        print("Warning: these extra columns were not found and will be skipped:")
        print(missing_extra_cols)

    selected_cols = [id_col, text_col] + valid_extra_cols
    cols_sql = ", ".join([f'"{col}"' for col in selected_cols])

    query = f'SELECT {cols_sql} FROM "{table_name}"'

    if where_clause:
        query += f" WHERE {where_clause}"

    conn = sqlite3.connect(db_path)

    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    return df


def predict_dataframe(
    pipe: Pipeline,
    df: pd.DataFrame,
    text_col: str,
    id_col: str,
) -> pd.DataFrame:
    texts = normalize_text(df[text_col])

    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(texts)

        pred_idx = probs.argmax(axis=1)
        predicted_labels = pipe.classes_[pred_idx]
        confidence = probs.max(axis=1)

        result = df.copy()

        if id_col != "comment_id":
            result = result.rename(columns={id_col: "comment_id"})

        result["predicted_agent"] = predicted_labels
        result["prediction_confidence"] = confidence

        for i, label in enumerate(pipe.classes_):
            result[f"prob_{label}"] = probs[:, i]

    else:
        predicted_labels = pipe.predict(texts)

        result = df.copy()

        if id_col != "comment_id":
            result = result.rename(columns={id_col: "comment_id"})

        result["predicted_agent"] = predicted_labels
        result["prediction_confidence"] = np.nan

    return result


def write_predictions_to_sqlite(
    df: pd.DataFrame,
    db_path: Path,
    output_table: str,
) -> None:
    conn = sqlite3.connect(db_path)

    try:
        df.to_sql(
            output_table,
            conn,
            if_exists="replace",
            index=False
        )
    finally:
        conn.close()


# =========================================================
# Main prediction command
# =========================================================

def cmd_predict_db(args):
    model_path = Path(args.model_file)
    input_db = Path(args.db_file)
    output_db = Path(args.output_db)

    print(f"Loading model: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not input_db.exists():
        raise FileNotFoundError(f"Input database not found: {input_db}")

    pipe = joblib.load(model_path)

    extra_cols = []

    if args.extra_cols:
        extra_cols = [
            col.strip()
            for col in args.extra_cols.split(",")
            if col.strip()
        ]

    print(f"Reading comments from database: {input_db}")
    print(f"Source table: {args.table_name}")

    df = read_sqlite_comments(
        db_path=input_db,
        table_name=args.table_name,
        id_col=args.id_col,
        text_col=args.text_col,
        extra_cols=extra_cols,
        where_clause=args.where,
    )

    print("Rows loaded:", len(df))

    df = df[df[args.text_col].fillna("").astype(str).str.strip() != ""].copy()

    print("Rows after removing empty text:", len(df))

    if len(df) == 0:
        raise ValueError("No valid rows to classify.")

    print("Running predictions...")

    result = predict_dataframe(
        pipe=pipe,
        df=df,
        text_col=args.text_col,
        id_col=args.id_col,
    )

    print(f"Writing predictions to database: {output_db}")
    print(f"Output table: {args.output_table}")

    write_predictions_to_sqlite(
        df=result,
        db_path=output_db,
        output_table=args.output_table,
    )

    preview = (
        result
        .head(args.preview_rows)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    classes = list(pipe.classes_) if hasattr(pipe, "classes_") else []

    print_json({
        "status": "ok",
        "command": "predict-db",
        "rows_scored": int(len(result)),
        "source_db": str(input_db),
        "source_table": args.table_name,
        "output_db": str(output_db),
        "output_table": args.output_table,
        "classes": classes,
        "preview_rows": preview,
    })


# =========================================================
# CLI
# =========================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Classifica i commenti usando un modello joblib già addestrato "
            "e salva le predizioni in SQLite."
        )
    )

    sub = parser.add_subparsers(dest="command", required=True)

    predict_parser = sub.add_parser(
        "predict-db",
        help="Classifica commenti dentro un database SQLite"
    )

    predict_parser.add_argument(
        "--model-file",
        default=str(DEFAULT_MODEL_PATH),
        help="Modello .joblib già addestrato"
    )

    predict_parser.add_argument(
        "--db-file",
        default=str(DEFAULT_INPUT_DB),
        help="Database sorgente"
    )

    predict_parser.add_argument(
        "--table-name",
        default="comments",
        help="Tabella sorgente"
    )

    predict_parser.add_argument(
        "--id-col",
        default="id",
        help="Colonna ID nella tabella comments"
    )

    predict_parser.add_argument(
        "--text-col",
        default="body",
        help="Colonna testo nella tabella comments"
    )

    predict_parser.add_argument(
        "--extra-cols",
        default="post_id,parent_comment_id,depth,comment_order,author,author_link,scraped_at",
        help="Altre colonne da copiare nell'output, separate da virgola"
    )

    predict_parser.add_argument(
        "--where",
        default=None,
        help="Filtro SQL opzionale"
    )

    predict_parser.add_argument(
        "--output-db",
        default=str(DEFAULT_OUTPUT_DB),
        help="Database SQLite di output"
    )

    predict_parser.add_argument(
        "--output-table",
        default=DEFAULT_SQLITE_OUTPUT_TABLE,
        help="Tabella di output"
    )

    predict_parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Numero di righe di preview stampate a terminale"
    )

    predict_parser.set_defaults(func=cmd_predict_db)

    return parser


def main():
    try:
        parser = build_arg_parser()
        args = parser.parse_args()
        args.func(args)

    except Exception as e:
        print_json({
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
        }, exit_code=1)


if __name__ == "__main__":
    main()