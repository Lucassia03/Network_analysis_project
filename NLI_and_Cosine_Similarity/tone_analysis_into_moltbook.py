import argparse
import sqlite3
import re
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np


# ==================================================
# CONFIG
# ==================================================

DEFAULT_DB = "moltbook_final_NLI_sim_WITH_CLUSTER_AGENT_MAPPING.db"
DEFAULT_OUTPUT_TABLE = "tone_analysis"


# ==================================================
# TONE MARKERS
# ==================================================

POSITIVE_MARKERS = [
    r"\bgreat\b",
    r"\bgood\b",
    r"\bexcellent\b",
    r"\bamazing\b",
    r"\bawesome\b",
    r"\bbrilliant\b",
    r"\bsmart\b",
    r"\binsightful\b",
    r"\bhelpful\b",
    r"\buseful\b",
    r"\bvaluable\b",
    r"\binteresting\b",
    r"\bnice\b",
    r"\blove\b",
    r"\bloved\b",
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bappreciate\b",
]

NEGATIVE_MARKERS = [
    r"\bbad\b",
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bfalse\b",
    r"\bterrible\b",
    r"\bawful\b",
    r"\bstupid\b",
    r"\bdumb\b",
    r"\bweak\b",
    r"\bflawed\b",
    r"\bproblem\b",
    r"\bissue\b",
    r"\bconcern\b",
    r"\bmisleading\b",
    r"\bconfusing\b",
    r"\buseless\b",
    r"\bnot helpful\b",
]

AGREEMENT_MARKERS = [
    r"\bi agree\b",
    r"\bagree\b",
    r"\bagreed\b",
    r"\bexactly\b",
    r"\babsolutely\b",
    r"\bdefinitely\b",
    r"\byes\b",
    r"\byeah\b",
    r"\byep\b",
    r"\btrue\b",
    r"\bcorrect\b",
    r"\bright\b",
    r"\byou'?re right\b",
    r"\byou are right\b",
    r"\bthat makes sense\b",
    r"\bmakes sense\b",
    r"\bfair enough\b",
]

DISAGREEMENT_MARKERS = [
    r"\bi disagree\b",
    r"\bdisagree\b",
    r"\bnot necessarily\b",
    r"\bnot really\b",
    r"\bno\b",
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bfalse\b",
    r"\bi don'?t think\b",
    r"\bi do not think\b",
    r"\bhowever\b",
    r"\bbut\b",
    r"\balthough\b",
    r"\bthough\b",
    r"\bcounterpoint\b",
    r"\bon the other hand\b",
]

CERTAINTY_MARKERS = [
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bobviously\b",
    r"\bclearly\b",
    r"\bwithout doubt\b",
    r"\bno doubt\b",
    r"\bmust\b",
    r"\balways\b",
    r"\bnever\b",
    r"\bguaranteed\b",
]

UNCERTAINTY_MARKERS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\bi think\b",
    r"\bi guess\b",
    r"\bi suppose\b",
    r"\bnot sure\b",
    r"\bunclear\b",
    r"\bseems\b",
    r"\bappears\b",
    r"\bmight\b",
    r"\bcould\b",
]

POLITENESS_MARKERS = [
    r"\bplease\b",
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bappreciate\b",
    r"\bsorry\b",
    r"\bexcuse me\b",
    r"\bif you don'?t mind\b",
]


# ==================================================
# SQLITE HELPERS
# ==================================================

def resolve_db_path(db_arg: str) -> Path:
    path = Path(db_arg)

    if path.exists():
        return path

    if path.suffix != ".db":
        path_with_db = Path(str(path) + ".db")
        if path_with_db.exists():
            return path_with_db

    raise FileNotFoundError(
        f"Database not found: {db_arg}. "
        "Check that the file exists in the current directory."
    )


def list_tables(db_path: Path) -> List[str]:
    conn = sqlite3.connect(db_path)

    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name;
            """
        ).fetchall()

        return [row[0] for row in rows]

    finally:
        conn.close()


def get_table_columns(db_path: Path, table_name: str) -> List[str]:
    conn = sqlite3.connect(db_path)

    try:
        rows = conn.execute(
            f'PRAGMA table_info("{table_name}");'
        ).fetchall()

        return [row[1] for row in rows]

    finally:
        conn.close()


def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    try:
        return pd.read_sql_query(
            f'SELECT * FROM "{table_name}";',
            conn
        )

    finally:
        conn.close()


def save_table(db_path: Path, table_name: str, df: pd.DataFrame) -> None:
    conn = sqlite3.connect(db_path)

    try:
        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

    finally:
        conn.close()


def find_source_table(
    db_path: Path,
    preferred_table: Optional[str] = None
) -> str:
    """
    Finds a source table containing the required 'body' column.
    """

    tables = list_tables(db_path)

    if preferred_table is not None:
        if preferred_table not in tables:
            raise ValueError(
                f"Table '{preferred_table}' not found. "
                f"Available tables: {tables}"
            )

        columns = get_table_columns(db_path, preferred_table)

        if "body" not in columns:
            raise KeyError(
                f"Table '{preferred_table}' does not contain required column 'body'. "
                f"Columns found: {columns}"
            )

        return preferred_table

    preferred_names = [
        "comments",
        "comment_cluster_agent_mapping",
        "moltbook_final_NLI_sim_WITH_CLUSTER_AGENT_MAPPING",
        "parent_child_similarity_with_cluster_agent_mapping",
    ]

    for table in preferred_names:
        if table in tables:
            columns = get_table_columns(db_path, table)

            if "body" in columns:
                return table

    for table in tables:
        columns = get_table_columns(db_path, table)

        if "body" in columns:
            return table

    raise KeyError(
        "No table with required column 'body' found. "
        f"Available tables: {tables}"
    )


# ==================================================
# TEXT / TONE HELPERS
# ==================================================

def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def count_markers(text: object, patterns: List[str]) -> int:
    text = normalize_text(text)

    total = 0

    for pattern in patterns:
        total += len(re.findall(pattern, text, flags=re.IGNORECASE))

    return total


def score_markers(text: object, patterns: List[str]) -> float:
    """
    Normalized score:
        0 markers  -> 0.00
        1 marker   -> 0.33
        2 markers  -> 0.67
        3+ markers -> 1.00
    """

    return min(count_markers(text, patterns) / 3.0, 1.0)


def word_count(text: object) -> int:
    text = normalize_text(text)

    return len(re.findall(r"\b\w+\b", text))


def compute_tone_label(row: pd.Series) -> str:
    positive = row["positive_score"]
    negative = row["negative_score"]
    agreement = row["agreement_score"]
    disagreement = row["disagreement_score"]
    certainty = row["certainty_score"]
    uncertainty = row["uncertainty_score"]
    politeness = row["politeness_score"]

    if negative >= 0.34 and disagreement >= 0.34:
        return "critical_disagreeing"

    if positive >= 0.34 and agreement >= 0.34:
        return "positive_agreeing"

    if positive >= 0.34 and politeness >= 0.34:
        return "polite_positive"

    if negative >= 0.34:
        return "negative"

    if disagreement >= 0.34:
        return "disagreeing"

    if agreement >= 0.34:
        return "agreeing"

    if uncertainty >= 0.34 and certainty < 0.34:
        return "uncertain"

    if certainty >= 0.34 and uncertainty < 0.34:
        return "assertive"

    if politeness >= 0.34:
        return "polite"

    if positive >= 0.34:
        return "positive"

    return "neutral_or_unclear"


# ==================================================
# TONE ANALYSIS
# ==================================================

def build_tone_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates one tone-analysis row per original row.

    Required:
        body

    Optional metadata preserved if present:
        id
        comment_id
        post_id
        author
        cluster
        cluster_name
        predicted_agent
        prediction_confidence
    """

    if "body" not in df.columns:
        raise KeyError("Input dataframe must contain column 'body'.")

    out = pd.DataFrame()

    if "comment_id" in df.columns:
        out["comment_id"] = df["comment_id"]
    elif "id" in df.columns:
        out["comment_id"] = df["id"]
    else:
        out["comment_id"] = np.arange(len(df))

    optional_columns = [
        "post_id",
        "author",
        "cluster",
        "cluster_name",
        "predicted_agent",
        "prediction_confidence",
    ]

    for col in optional_columns:
        if col in df.columns:
            out[col] = df[col]

    out["body"] = df["body"].fillna("").astype(str)

    out["word_count"] = out["body"].apply(word_count)

    out["positive_marker_count"] = out["body"].apply(
        lambda x: count_markers(x, POSITIVE_MARKERS)
    )

    out["negative_marker_count"] = out["body"].apply(
        lambda x: count_markers(x, NEGATIVE_MARKERS)
    )

    out["agreement_marker_count"] = out["body"].apply(
        lambda x: count_markers(x, AGREEMENT_MARKERS)
    )

    out["disagreement_marker_count"] = out["body"].apply(
        lambda x: count_markers(x, DISAGREEMENT_MARKERS)
    )

    out["certainty_marker_count"] = out["body"].apply(
        lambda x: count_markers(x, CERTAINTY_MARKERS)
    )

    out["uncertainty_marker_count"] = out["body"].apply(
        lambda x: count_markers(x, UNCERTAINTY_MARKERS)
    )

    out["politeness_marker_count"] = out["body"].apply(
        lambda x: count_markers(x, POLITENESS_MARKERS)
    )

    out["positive_score"] = out["body"].apply(
        lambda x: score_markers(x, POSITIVE_MARKERS)
    )

    out["negative_score"] = out["body"].apply(
        lambda x: score_markers(x, NEGATIVE_MARKERS)
    )

    out["agreement_score"] = out["body"].apply(
        lambda x: score_markers(x, AGREEMENT_MARKERS)
    )

    out["disagreement_score"] = out["body"].apply(
        lambda x: score_markers(x, DISAGREEMENT_MARKERS)
    )

    out["certainty_score"] = out["body"].apply(
        lambda x: score_markers(x, CERTAINTY_MARKERS)
    )

    out["uncertainty_score"] = out["body"].apply(
        lambda x: score_markers(x, UNCERTAINTY_MARKERS)
    )

    out["politeness_score"] = out["body"].apply(
        lambda x: score_markers(x, POLITENESS_MARKERS)
    )

    out["tone_label"] = out.apply(compute_tone_label, axis=1)

    return out


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simple tone analysis on the 'body' column and save it inside the same SQLite DB."
    )

    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help="SQLite database path."
    )

    parser.add_argument(
        "--table",
        default=None,
        help="Source table name. If omitted, the script auto-detects a table with 'body'."
    )

    parser.add_argument(
        "--output-table",
        default=DEFAULT_OUTPUT_TABLE,
        help="Output table name."
    )

    parser.add_argument(
        "--print-rows",
        type=int,
        default=50,
        help="Number of rows to print from the output table."
    )

    args = parser.parse_args()

    db_path = resolve_db_path(args.db)

    print("\nUsing database:")
    print(db_path)

    print("\nAvailable tables:")
    for table in list_tables(db_path):
        print("-", table)

    source_table = find_source_table(
        db_path=db_path,
        preferred_table=args.table
    )

    print("\nSource table selected:")
    print(source_table)

    df = load_table(db_path, source_table)

    print("\nRows loaded:")
    print(len(df))

    print("\nColumns loaded:")
    print(list(df.columns))

    print("\nRunning tone analysis on column: body")

    tone_df = build_tone_analysis(df)

    save_table(
        db_path=db_path,
        table_name=args.output_table,
        df=tone_df
    )

    print("\nSaved new table inside same database:")
    print(args.output_table)

    print("\nRows saved:")
    print(len(tone_df))

    print("\n" + "=" * 100)
    print("TONE ANALYSIS PREVIEW")
    print("=" * 100)

    print(
        tone_df
        .head(args.print_rows)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("TONE LABEL COUNTS")
    print("=" * 100)

    print(
        tone_df["tone_label"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
