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
DEFAULT_INPUT_DB = Path("moltbook_final.db")
DEFAULT_OUTPUT_DB = Path("agent_predictions_from_json.db")
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
    """
    Prints a JSON-like status message to the terminal only.
    It does NOT create a JSON file.
    """
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


def read_sqlite_comments(
    db_path: Path,
    table_name: str,
    id_col: str,
    text_col: str,
    extra_cols=None,
    where_clause: str | None = None,
) -> pd.DataFrame:
    extra_cols = extra_cols or []

    selected_cols = [id_col, text_col] + [
        c for c in extra_cols
        if c not in {id_col, text_col}
    ]

    cols_sql = ", ".join([f'"{c}"' for c in selected_cols])

    query = f'SELECT {cols_sql} FROM "{table_name}"'

    if where_clause:
        query += f" WHERE {where_clause}"

    conn = sqlite3.connect(db_path)

    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if id_col not in df.columns or text_col not in df.columns:
        raise KeyError(
            f"Colonne richieste non trovate. Colonne disponibili: {list(df.columns)}"
        )

    return df


def predict_dataframe(
    pipe: Pipeline,
    df: pd.DataFrame,
    text_col: str,
    id_col: str,
) -> pd.DataFrame:
    texts = normalize_text(df[text_col])

    probs = pipe.predict_proba(texts)

    pred_idx = probs.argmax(axis=1)
    predicted_labels = pipe.classes_[pred_idx]
    confidence = probs.max(axis=1)

    result = df.copy()

    # Downstream scripts expect comment_id.
    if id_col != "comment_id":
        result = result.rename(columns={id_col: "comment_id"})

    result["predicted_agent"] = predicted_labels
    result["prediction_confidence"] = confidence

    for i, label in enumerate(pipe.classes_):
        result[f"prob_{label}"] = probs[:, i]

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
    print(f"Loading model: {args.model_file}")

    pipe = joblib.load(args.model_file)

    extra_cols = []

    if args.extra_cols:
        extra_cols = [
            c.strip()
            for c in args.extra_cols.split(",")
            if c.strip()
        ]

    print(f"Reading comments from: {args.db_file}")

    df = read_sqlite_comments(
        db_path=Path(args.db_file),
        table_name=args.table_name,
        id_col=args.id_col,
        text_col=args.text_col,
        extra_cols=extra_cols,
        where_clause=args.where,
    )

    print("Rows loaded:", len(df))

    result = predict_dataframe(
        pipe=pipe,
        df=df,
        text_col=args.text_col,
        id_col=args.id_col,
    )

    print(f"Writing predictions to DB: {args.output_db}")
    print(f"Output table: {args.output_table}")

    write_predictions_to_sqlite(
        df=result,
        db_path=Path(args.output_db),
        output_table=args.output_table,
    )

    preview = (
        result
        .head(args.preview_rows)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    print_json({
        "status": "ok",
        "command": "predict-db",
        "rows_scored": int(len(result)),
        "source_db": str(args.db_file),
        "source_table": args.table_name,
        "output_db": str(args.output_db),
        "output_table": args.output_table,
        "classes": list(pipe.classes_),
        "preview_rows": preview,
    })


# =========================================================
# CLI
# =========================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Classifica i commenti di moltbook_final.db usando "
            "un modello joblib già addestrato e salva solo in SQLite."
        )
    )

    sub = parser.add_subparsers(dest="command", required=True)

    predict_parser = sub.add_parser(
        "predict-db",
        help="Classifica commenti dentro un DB SQLite"
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