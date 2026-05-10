import sqlite3
import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy.stats import chi2_contingency, chi2, kruskal
from sklearn.metrics import cohen_kappa_score


# ============================================================
# 1. PATHS
# ============================================================

db_path = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\moltbook_final_NLI_sim_1500_per_table (1).db"
)

output_dir = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\DATASET GAB"
)

output_dir.mkdir(parents=True, exist_ok=True)

if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

print("Using database:")
print(db_path)


# ============================================================
# 2. LOAD TABLES
# ============================================================

conn = sqlite3.connect(db_path)

tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn
)

print("\nAvailable tables:")
print(tables["name"].tolist())

comments = pd.read_sql_query("""
    SELECT
        id AS subset_comment_id,
        post_id,
        parent_comment_id,
        depth,
        comment_order,
        author,
        body,
        target_context,
        context_type,
        context_confidence,
        Final_context_extracted,
        LLM_Stance_detection,
        LLM_Stance_detection_explanation
    FROM comments
""", conn)

nli = pd.read_sql_query("""
    SELECT
        rowid AS nli_rowid,
        relation_type,
        post_id,
        parent_id,
        child_id AS original_comment_id,
        parent_author,
        child_author,
        parent_text,
        child_text,
        child_depth,
        similarity,
        contradiction_score,
        neutral_score,
        entailment_score,
        prediction AS nli_prediction
    FROM parent_child_similarity
""", conn)

conn.close()

print("\nRows in comments:", len(comments))
print("Rows in parent_child_similarity:", len(nli))


# ============================================================
# 3. CHECK ID STRUCTURE
# ============================================================

print("\nID ranges:")
print("comments.subset_comment_id min:", comments["subset_comment_id"].min())
print("comments.subset_comment_id max:", comments["subset_comment_id"].max())
print("nli.original_comment_id min:", nli["original_comment_id"].min())
print("nli.original_comment_id max:", nli["original_comment_id"].max())

id_merge_check = comments.merge(
    nli,
    left_on="subset_comment_id",
    right_on="original_comment_id",
    how="inner",
    suffixes=("_comments", "_nli")
)

print("\nMatches using subset_comment_id = original_comment_id:")
print(len(id_merge_check))

print(
    "\nShare of comments matched by ID:",
    round(100 * len(id_merge_check) / len(comments), 2),
    "%"
)

print(
    "Share of NLI rows matched by ID:",
    round(100 * len(id_merge_check) / len(nli), 2),
    "%"
)

print(
    "\nIMPORTANT: if this is much lower than 100%, "
    "comments.id and parent_child_similarity.child_id are not the same identifier."
)


# ============================================================
# 4. NORMALIZE TEXT FOR SAFE MATCHING
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x


comments["body_norm"] = comments["body"].apply(normalize_text)
nli["child_text_norm"] = nli["child_text"].apply(normalize_text)

comments["author_norm"] = comments["author"].fillna("").astype(str).str.lower().str.strip()
nli["child_author_norm"] = nli["child_author"].fillna("").astype(str).str.lower().str.strip()

comments["depth_norm"] = comments["depth"].fillna(-999).astype(int)
nli["child_depth_norm"] = nli["child_depth"].fillna(-999).astype(int)


# ============================================================
# 5. SAFE MERGE ON POST + TEXT + AUTHOR + DEPTH
# ============================================================

df = comments.merge(
    nli,
    left_on=[
        "post_id",
        "body_norm",
        "author_norm",
        "depth_norm"
    ],
    right_on=[
        "post_id",
        "child_text_norm",
        "child_author_norm",
        "child_depth_norm"
    ],
    how="inner",
    suffixes=("_comments", "_nli")
)

print("\nMatches using post_id + text + author + depth:")
print(len(df))

print(
    "Share of comments matched safely:",
    round(100 * len(df) / len(comments), 2),
    "%"
)

print(
    "Share of NLI rows matched safely:",
    round(100 * len(df) / len(nli), 2),
    "%"
)


# ============================================================
# 6. CHECK FOR DUPLICATES AFTER SAFE MERGE
# ============================================================

duplicated_comments = df["subset_comment_id"].duplicated().sum()
duplicated_nli = df["nli_rowid"].duplicated().sum()

print("\nDuplicate checks after safe merge:")
print("Duplicated subset_comment_id:", duplicated_comments)
print("Duplicated nli_rowid:", duplicated_nli)

if duplicated_comments > 0 or duplicated_nli > 0:
    print(
        "\nWARNING: The merge is not one-to-one. "
        "You need to inspect duplicates before doing validation."
    )

    duplicate_comment_rows = df[
        df["subset_comment_id"].duplicated(keep=False)
    ].sort_values("subset_comment_id")

    duplicate_nli_rows = df[
        df["nli_rowid"].duplicated(keep=False)
    ].sort_values("nli_rowid")

    duplicate_comment_rows.to_csv(
        output_dir / "duplicate_comments_after_safe_merge.csv",
        index=False
    )

    duplicate_nli_rows.to_csv(
        output_dir / "duplicate_nli_rows_after_safe_merge.csv",
        index=False
    )

    raise ValueError(
        "Safe merge created duplicates. Inspect duplicate CSV files before continuing."
    )


# ============================================================
# 7. COMPARE TEXTS AFTER MERGE
# ============================================================

df["text_exact_match"] = (
    df["body_norm"] == df["child_text_norm"]
).astype(int)

df["author_exact_match"] = (
    df["author_norm"] == df["child_author_norm"]
).astype(int)

df["depth_exact_match"] = (
    df["depth_norm"] == df["child_depth_norm"]
).astype(int)

print("\nPost-merge validation checks:")
print("Text exact match rate:", round(100 * df["text_exact_match"].mean(), 2), "%")
print("Author exact match rate:", round(100 * df["author_exact_match"].mean(), 2), "%")
print("Depth exact match rate:", round(100 * df["depth_exact_match"].mean(), 2), "%")

if df["text_exact_match"].mean() < 1:
    print("\nExamples of text mismatch:")
    print(
        df.loc[
            df["text_exact_match"] == 0,
            [
                "subset_comment_id",
                "original_comment_id",
                "post_id",
                "body",
                "child_text"
            ]
        ].head(10)
    )


# ============================================================
# 8. SAVE MATCH AUDIT
# ============================================================

match_audit = pd.DataFrame([{
    "comments_rows": len(comments),
    "nli_rows": len(nli),
    "id_merge_matches": len(id_merge_check),
    "safe_merge_matches": len(df),
    "id_merge_share_comments_pct": 100 * len(id_merge_check) / len(comments),
    "safe_merge_share_comments_pct": 100 * len(df) / len(comments),
    "duplicated_comments_after_safe_merge": duplicated_comments,
    "duplicated_nli_after_safe_merge": duplicated_nli,
    "text_exact_match_rate": df["text_exact_match"].mean(),
    "author_exact_match_rate": df["author_exact_match"].mean(),
    "depth_exact_match_rate": df["depth_exact_match"].mean()
}])

match_audit.to_csv(
    output_dir / "nli_llm_safe_merge_audit.csv",
    index=False
)

df.to_csv(
    output_dir / "nli_llm_safely_matched_dataset.csv",
    index=False
)

print("\nSaved merge audit:")
print(output_dir / "nli_llm_safe_merge_audit.csv")

print("\nSaved safely matched dataset:")
print(output_dir / "nli_llm_safely_matched_dataset.csv")


# ============================================================
# 9. CLEAN LABELS
# ============================================================

def clean_label(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    if x in ["entailment", "entailed", "support", "supports"]:
        return "entailment"

    if x in ["neutral", "neither", "unclear", "ambiguous"]:
        return "neutral"

    if x in ["contradiction", "contradicts", "contradict"]:
        return "contradiction"

    return np.nan


df["nli_label"] = df["nli_prediction"].apply(clean_label)
df["llm_label"] = df["LLM_Stance_detection"].apply(clean_label)

print("\nRaw NLI labels:")
print(df["nli_prediction"].value_counts(dropna=False))

print("\nRaw LLM stance labels:")
print(df["LLM_Stance_detection"].value_counts(dropna=False))

df = df.dropna(subset=["nli_label", "llm_label"]).copy()

label_order = ["contradiction", "neutral", "entailment"]

df["nli_label"] = pd.Categorical(
    df["nli_label"],
    categories=label_order,
    ordered=True
)

df["llm_label"] = pd.Categorical(
    df["llm_label"],
    categories=label_order,
    ordered=True
)

print("\nCleaned observations:", len(df))

print("\nCleaned NLI labels:")
print(df["nli_label"].value_counts())

print("\nCleaned LLM labels:")
print(df["llm_label"].value_counts())


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

conf_table = pd.crosstab(
    df["nli_label"],
    df["llm_label"],
    rownames=["NLI_prediction"],
    colnames=["LLM_as_judge"]
)

conf_table = conf_table.reindex(
    index=label_order,
    columns=label_order,
    fill_value=0
)

print("\nConfusion matrix: NLI prediction x LLM-as-judge")
print(conf_table)

conf_table.to_csv(
    output_dir / "nli_vs_llm_confusion_matrix_SAFE_ID_MATCH.csv"
)


# ============================================================
# 11. ROW AND COLUMN PERCENTAGES
# ============================================================

row_pct = conf_table.div(conf_table.sum(axis=1), axis=0) * 100
col_pct = conf_table.div(conf_table.sum(axis=0), axis=1) * 100

print("\nRow percentages: within NLI prediction")
print(row_pct.round(2))

print("\nColumn percentages: within LLM-as-judge")
print(col_pct.round(2))

row_pct.to_csv(
    output_dir / "nli_vs_llm_row_percentages_SAFE_ID_MATCH.csv"
)

col_pct.to_csv(
    output_dir / "nli_vs_llm_column_percentages_SAFE_ID_MATCH.csv"
)


# ============================================================
# 12. EXACT AGREEMENT
# ============================================================

df["exact_agreement"] = (
    df["nli_label"].astype(str) == df["llm_label"].astype(str)
).astype(int)

agreement_rate = df["exact_agreement"].mean()

print("\nExact agreement rate:")
print(round(agreement_rate * 100, 2), "%")


# ============================================================
# 13. COHEN'S KAPPA
# ============================================================

y_nli = df["nli_label"].astype(str)
y_llm = df["llm_label"].astype(str)

kappa_unweighted = cohen_kappa_score(
    y_nli,
    y_llm,
    labels=label_order
)

kappa_linear = cohen_kappa_score(
    y_nli,
    y_llm,
    labels=label_order,
    weights="linear"
)

kappa_quadratic = cohen_kappa_score(
    y_nli,
    y_llm,
    labels=label_order,
    weights="quadratic"
)

print("\nCohen's kappa:")
print("Unweighted kappa:", round(kappa_unweighted, 4))
print("Linear weighted kappa:", round(kappa_linear, 4))
print("Quadratic weighted kappa:", round(kappa_quadratic, 4))

kappa_results = pd.DataFrame([{
    "n": len(df),
    "exact_agreement_rate": agreement_rate,
    "unweighted_kappa": kappa_unweighted,
    "linear_weighted_kappa": kappa_linear,
    "quadratic_weighted_kappa": kappa_quadratic
}])

kappa_results.to_csv(
    output_dir / "nli_vs_llm_kappa_results_SAFE_ID_MATCH.csv",
    index=False
)


# ============================================================
# 14. CHI-SQUARE TEST OF ASSOCIATION
# ============================================================

chi2_stat, chi2_p, chi2_dof, expected = chi2_contingency(conf_table)

cramers_v = np.sqrt(
    chi2_stat / (
        conf_table.values.sum()
        * min(conf_table.shape[0] - 1, conf_table.shape[1] - 1)
    )
)

print("\nChi-square test of association")
print("Chi-square:", round(chi2_stat, 4))
print("Degrees of freedom:", chi2_dof)
print("p-value:", chi2_p)
print("Cramer's V:", round(cramers_v, 4))

chi_results = pd.DataFrame([{
    "test": "Chi-square association test",
    "chi2": chi2_stat,
    "dof": chi2_dof,
    "p_value": chi2_p,
    "cramers_v": cramers_v,
    "n": conf_table.values.sum()
}])

chi_results.to_csv(
    output_dir / "nli_vs_llm_chi_square_SAFE_ID_MATCH.csv",
    index=False
)


# ============================================================
# 15. STANDARDIZED RESIDUALS
# ============================================================

expected_df = pd.DataFrame(
    expected,
    index=conf_table.index,
    columns=conf_table.columns
)

std_residuals = (conf_table - expected_df) / np.sqrt(expected_df)

print("\nStandardized residuals:")
print(std_residuals.round(3))

std_residuals.to_csv(
    output_dir / "nli_vs_llm_standardized_residuals_SAFE_ID_MATCH.csv"
)


# ============================================================
# 16. BOWKER'S TEST OF SYMMETRY
# ============================================================

def bowker_test_square_table(table):
    arr = np.asarray(table, dtype=float)

    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Bowker test requires a square table.")

    stat = 0.0
    dof = 0

    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[1]):
            nij = arr[i, j]
            nji = arr[j, i]

            if nij + nji > 0:
                stat += (nij - nji) ** 2 / (nij + nji)
                dof += 1

    p_value = 1 - chi2.cdf(stat, dof)

    return stat, dof, p_value


bowker_stat, bowker_dof, bowker_p = bowker_test_square_table(conf_table)

print("\nBowker's test of symmetry")
print("Statistic:", round(bowker_stat, 4))
print("Degrees of freedom:", bowker_dof)
print("p-value:", bowker_p)

bowker_results = pd.DataFrame([{
    "test": "Bowker symmetry test",
    "statistic": bowker_stat,
    "dof": bowker_dof,
    "p_value": bowker_p,
    "n": conf_table.values.sum()
}])

bowker_results.to_csv(
    output_dir / "nli_vs_llm_bowker_SAFE_ID_MATCH.csv",
    index=False
)


# ============================================================
# 17. DISAGREEMENT SUMMARY
# ============================================================

disagreement_summary = (
    df
    .groupby(["nli_label", "llm_label"], observed=False)
    .agg(
        n=("subset_comment_id", "count"),
        mean_similarity=("similarity", "mean"),
        mean_contradiction_score=("contradiction_score", "mean"),
        mean_neutral_score=("neutral_score", "mean"),
        mean_entailment_score=("entailment_score", "mean"),
        mean_context_confidence=("context_confidence", "mean")
    )
    .reset_index()
)

disagreement_summary["pct_total"] = (
    100 * disagreement_summary["n"] / len(df)
)

print("\nDisagreement / agreement type summary:")
print(disagreement_summary.round(4))

disagreement_summary.to_csv(
    output_dir / "nli_vs_llm_disagreement_summary_SAFE_ID_MATCH.csv",
    index=False
)


# ============================================================
# 18. AGREEMENT BY CONTEXT TYPE
# ============================================================

context_summary = (
    df
    .groupby("context_type")
    .agg(
        n=("subset_comment_id", "count"),
        agreement_rate=("exact_agreement", "mean"),
        mean_similarity=("similarity", "mean"),
        mean_context_confidence=("context_confidence", "mean")
    )
    .reset_index()
    .sort_values("agreement_rate", ascending=False)
)

context_summary["agreement_rate_pct"] = 100 * context_summary["agreement_rate"]

print("\nAgreement by context_type:")
print(context_summary.round(4))

context_summary.to_csv(
    output_dir / "nli_vs_llm_agreement_by_context_type_SAFE_ID_MATCH.csv",
    index=False
)


# ============================================================
# 19. SAVE FINAL VALIDATION DATASET
# ============================================================

df.to_csv(
    output_dir / "nli_vs_llm_validation_dataset_SAFE_ID_MATCH.csv",
    index=False
)

print("\nSaved final validation dataset:")
print(output_dir / "nli_vs_llm_validation_dataset_SAFE_ID_MATCH.csv")