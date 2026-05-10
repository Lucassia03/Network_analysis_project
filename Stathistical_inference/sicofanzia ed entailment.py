import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# ============================================================
# 1. PATHS
# ============================================================

model_db_path = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\DATASET GAB\moltbook_final_NLI_sim_WITH_CLUSTER_AGENT_MAPPING 3.db"
)

syc_db_path = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\DATASET GAB\moltbook_merged_4_dbs_all_tables.db"
)

output_dir = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\DATASET GAB"
)

output_dir.mkdir(parents=True, exist_ok=True)

print("Model DB exists:", model_db_path.exists())
print("Sycophancy DB exists:", syc_db_path.exists())

if not model_db_path.exists():
    raise FileNotFoundError(f"Model DB not found: {model_db_path}")

if not syc_db_path.exists():
    raise FileNotFoundError(f"Sycophancy DB not found: {syc_db_path}")


# ============================================================
# 2. LOAD MODEL TYPE FROM MAPPING 3 DB
# ============================================================

conn_model = sqlite3.connect(model_db_path)

# Check available tables
model_tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn_model
)

print("\nTables in model DB:")
print(model_tables["name"].tolist())

# Questo è il dataset nuovo da cui prendiamo il modello.
# Se il tuo DB ha il modello in una tabella con nome diverso,
# lo vedi stampato sopra.
model = pd.read_sql_query("""
    SELECT
        id AS comment_id,
        predicted_model,
        prediction_confidence,
        author,
        depth,
        comment_order,
        parent_comment_id
    FROM comments_model_predictions
""", conn_model)

conn_model.close()

print("\nModel observations:", len(model))
print("\nPredicted model counts before cleaning:")
print(model["predicted_model"].value_counts(dropna=False))


# ============================================================
# 3. LOAD SYCOPHANCY DATA FROM MERGED DB
# ============================================================

conn_syc = sqlite3.connect(syc_db_path)

syc_tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn_syc
)

print("\nTables in sycophancy DB:")
print(syc_tables["name"].tolist())

syc = pd.read_sql_query("""
    SELECT
        comment_id,
        sycophancy_keyword_count,
        has_sycophancy_keyword
    FROM comment_clusters
""", conn_syc)

conn_syc.close()

print("\nSycophancy observations:", len(syc))


# ============================================================
# 4. MERGE MODEL TYPE + SYCOPHANCY
# ============================================================

model["comment_id"] = model["comment_id"].astype(str)
syc["comment_id"] = syc["comment_id"].astype(str)

df = model.merge(
    syc,
    on="comment_id",
    how="inner"
)

print("\nMerged observations:", len(df))

print("\nPredicted model counts after merge:")
print(df["predicted_model"].value_counts(dropna=False))

# Keep only comments with predicted model
df = df.dropna(subset=["predicted_model"]).copy()

print("\nMerged observations with predicted_model:", len(df))
print("\nNumber of predicted models:", df["predicted_model"].nunique())

df["sycophancy_keyword_count"] = (
    df["sycophancy_keyword_count"]
    .fillna(0)
    .astype(int)
)

df["has_sycophancy_keyword"] = (
    df["sycophancy_keyword_count"] > 0
).astype(int)

df["sycophancy_2plus"] = (
    df["sycophancy_keyword_count"] >= 2
).astype(int)

# Optional robustness: keep only confident predictions
# Decommenta se vuoi tenere solo predizioni più affidabili.
# df = df[df["prediction_confidence"] >= 0.60].copy()


# ============================================================
# 5. DESCRIPTIVE MODEL TABLE
# ============================================================

model_summary = (
    df
    .groupby("predicted_model")
    .agg(
        n_comments=("comment_id", "count"),
        mean_count=("sycophancy_keyword_count", "mean"),
        median_count=("sycophancy_keyword_count", "median"),
        q75_count=("sycophancy_keyword_count", lambda x: x.quantile(0.75)),
        q90_count=("sycophancy_keyword_count", lambda x: x.quantile(0.90)),
        pct_any_sycophancy=("has_sycophancy_keyword", lambda x: 100 * x.mean()),
        pct_2plus_keywords=("sycophancy_2plus", lambda x: 100 * x.mean()),
        avg_prediction_confidence=("prediction_confidence", "mean"),
        median_prediction_confidence=("prediction_confidence", "median")
    )
    .reset_index()
    .sort_values("pct_any_sycophancy", ascending=False)
)

print("\nModel-level sycophancy summary:")
print(model_summary.round(3))

model_summary.to_csv(
    output_dir / "nonparametric_model_sycophancy_summary.csv",
    index=False
)


# ============================================================
# 6. KRUSKAL-WALLIS TEST
# ============================================================

groups = [
    group["sycophancy_keyword_count"].values
    for _, group in df.groupby("predicted_model")
]

if len(groups) < 2:
    raise ValueError("Kruskal-Wallis requires at least two predicted_model groups.")

kw = kruskal(*groups)

n = len(df)
k = df["predicted_model"].nunique()

epsilon_squared = (kw.statistic - k + 1) / (n - k)
epsilon_squared = max(0, epsilon_squared)

print("\nKruskal-Wallis test")
print("Grouping variable: predicted_model")
print("Outcome: sycophancy_keyword_count")
print("H statistic:", round(kw.statistic, 3))
print("p-value:", kw.pvalue)
print("epsilon-squared:", round(epsilon_squared, 4))

kw_results = pd.DataFrame([{
    "test": "Kruskal-Wallis",
    "group_var": "predicted_model",
    "outcome": "sycophancy_keyword_count",
    "H_statistic": kw.statistic,
    "p_value": kw.pvalue,
    "epsilon_squared": epsilon_squared,
    "n": n,
    "k_groups": k
}])

kw_results.to_csv(
    output_dir / "nonparametric_model_sycophancy_kruskal.csv",
    index=False
)


# ============================================================
# 7. PAIRWISE MANN-WHITNEY U TESTS WITH FDR CORRECTION
# ============================================================

models = sorted(df["predicted_model"].dropna().unique())

pairwise_results = []

for m1, m2 in combinations(models, 2):
    x = df.loc[df["predicted_model"] == m1, "sycophancy_keyword_count"]
    y = df.loc[df["predicted_model"] == m2, "sycophancy_keyword_count"]

    u_stat, p_val = mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    )

    pairwise_results.append({
        "model_1": m1,
        "model_2": m2,
        "n_1": len(x),
        "n_2": len(y),
        "median_1": x.median(),
        "median_2": y.median(),
        "mean_1": x.mean(),
        "mean_2": y.mean(),
        "pct_any_1": 100 * (x > 0).mean(),
        "pct_any_2": 100 * (y > 0).mean(),
        "u_stat": u_stat,
        "p_value_raw": p_val
    })

pairwise_df = pd.DataFrame(pairwise_results)

if not pairwise_df.empty:
    reject, p_adj, _, _ = multipletests(
        pairwise_df["p_value_raw"],
        method="fdr_bh"
    )

    pairwise_df["p_value_fdr"] = p_adj
    pairwise_df["reject_fdr_5pct"] = reject

    pairwise_df = pairwise_df.sort_values("p_value_fdr")

    print("\nPairwise Mann-Whitney results:")
    print(pairwise_df.round(4))

    pairwise_df.to_csv(
        output_dir / "nonparametric_pairwise_model_sycophancy_mannwhitney_fdr.csv",
        index=False
    )


# ============================================================
# 8. CHI-SQUARE ON CATEGORIZED INTENSITY
# ============================================================

df["sycophancy_intensity"] = pd.cut(
    df["sycophancy_keyword_count"],
    bins=[-1, 0, 1, np.inf],
    labels=["none", "one_keyword", "two_or_more_keywords"]
)

intensity_table = pd.crosstab(
    df["predicted_model"],
    df["sycophancy_intensity"]
)

chi2, p_chi, dof, expected = chi2_contingency(intensity_table)

cramers_v = np.sqrt(
    chi2 / (
        intensity_table.values.sum()
        * min(
            intensity_table.shape[0] - 1,
            intensity_table.shape[1] - 1
        )
    )
)

print("\nChi-square test: model x sycophancy intensity category")
print(intensity_table)
print("Chi-square:", round(chi2, 3))
print("Degrees of freedom:", dof)
print("p-value:", p_chi)
print("Cramer's V:", round(cramers_v, 3))

intensity_table.to_csv(
    output_dir / "nonparametric_model_sycophancy_intensity_table.csv"
)

chi_results = pd.DataFrame([{
    "test": "Chi-square",
    "row_var": "predicted_model",
    "col_var": "sycophancy_intensity",
    "chi2": chi2,
    "dof": dof,
    "p_value": p_chi,
    "cramers_v": cramers_v
}])

chi_results.to_csv(
    output_dir / "nonparametric_model_sycophancy_chi_square.csv",
    index=False
)


# ============================================================
# 9. STANDARDIZED RESIDUALS
# ============================================================

expected_df = pd.DataFrame(
    expected,
    index=intensity_table.index,
    columns=intensity_table.columns
)

std_residuals = (intensity_table - expected_df) / np.sqrt(expected_df)

print("\nStandardized residuals:")
print(std_residuals.round(3))

std_residuals.to_csv(
    output_dir / "nonparametric_model_sycophancy_intensity_standardized_residuals.csv"
)


# ============================================================
# 10. OPTIONAL: PERMUTATION TEST
# ============================================================

def between_model_rank_variance(data, y_col, model_col):
    temp = data[[y_col, model_col]].copy()
    temp["rank_y"] = temp[y_col].rank(method="average")

    model_means = temp.groupby(model_col)["rank_y"].mean()
    model_sizes = temp.groupby(model_col)["rank_y"].size()
    grand_mean = temp["rank_y"].mean()

    return np.average(
        (model_means - grand_mean) ** 2,
        weights=model_sizes
    )


observed_stat = between_model_rank_variance(
    df,
    y_col="sycophancy_keyword_count",
    model_col="predicted_model"
)

rng = np.random.default_rng(123)
n_perm = 1000
perm_stats = []

for _ in range(n_perm):
    temp = df.copy()
    temp["predicted_model_perm"] = rng.permutation(
        temp["predicted_model"].values
    )

    perm_stats.append(
        between_model_rank_variance(
            temp,
            y_col="sycophancy_keyword_count",
            model_col="predicted_model_perm"
        )
    )

perm_stats = np.array(perm_stats)

perm_p = (np.sum(perm_stats >= observed_stat) + 1) / (n_perm + 1)

print("\nPermutation test on between-model rank variance")
print("Observed statistic:", observed_stat)
print("Permutation p-value:", perm_p)

perm_results = pd.DataFrame([{
    "test": "Permutation test",
    "group_var": "predicted_model",
    "outcome": "sycophancy_keyword_count",
    "observed_statistic": observed_stat,
    "n_permutations": n_perm,
    "p_value": perm_p
}])

perm_results.to_csv(
    output_dir / "nonparametric_model_sycophancy_permutation.csv",
    index=False
)


# ============================================================
# 11. SAVE FINAL DATASET
# ============================================================

df.to_csv(
    output_dir / "nonparametric_model_sycophancy_dataset.csv",
    index=False
)

print("\nSaved files:")
print(output_dir / "nonparametric_model_sycophancy_summary.csv")
print(output_dir / "nonparametric_model_sycophancy_kruskal.csv")
print(output_dir / "nonparametric_pairwise_model_sycophancy_mannwhitney_fdr.csv")
print(output_dir / "nonparametric_model_sycophancy_intensity_table.csv")
print(output_dir / "nonparametric_model_sycophancy_chi_square.csv")
print(output_dir / "nonparametric_model_sycophancy_intensity_standardized_residuals.csv")
print(output_dir / "nonparametric_model_sycophancy_permutation.csv")
print(output_dir / "nonparametric_model_sycophancy_dataset.csv")
