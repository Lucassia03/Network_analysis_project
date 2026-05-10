import sqlite3
import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# ============================================================
# 1. PATHS
# ============================================================

DB_PATH = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\moltbook_final_NLI_sim_WITH_CLUSTER_AGENT_MAPPING 3.db"
)

OUTPUT_DIR = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\DATASET GAB"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SICOPHANCY KEYWORDS
# ============================================================

SYCOPHANCY_KEYWORDS = [
    "absolutely",
    "exactly",
    "definitely",
    "completely",
    "totally",
    "perfectly",
    "certainly",
    "clearly",
    "obviously",
    "right",
    "correct",
    "true",
    "great",
    "excellent",
    "brilliant",
    "smart",
    "insightful",
    "thoughtful",
    "interesting",
    "good point",
    "great point",
    "great question",
    "you are right",
    "you're right",
    "youre right",
    "i agree",
    "agree",
    "makes sense",
    "that makes sense",
    "well said",
    "nicely put",
    "important point",
    "valid point",
    "thanks",
    "thank you",
    "appreciate",
    "helpful",
]


def count_sycophancy_keywords(text, keywords=SYCOPHANCY_KEYWORDS):
    """
    Counts occurrences of sycophancy-related keywords/phrases in a text.
    Uses word boundaries where possible to avoid partial matches.
    """
    if pd.isna(text):
        return 0

    text = str(text).lower()
    count = 0

    for kw in keywords:
        kw_lower = kw.lower().strip()

        # Phrase matching
        if " " in kw_lower or "'" in kw_lower:
            pattern = re.escape(kw_lower)
        else:
            pattern = r"\b" + re.escape(kw_lower) + r"\b"

        matches = re.findall(pattern, text)
        count += len(matches)

    return count


def matched_sycophancy_keywords(text, keywords=SYCOPHANCY_KEYWORDS):
    """
    Returns the list of sycophancy keywords found in a text.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    found = []

    for kw in keywords:
        kw_lower = kw.lower().strip()

        if " " in kw_lower or "'" in kw_lower:
            pattern = re.escape(kw_lower)
        else:
            pattern = r"\b" + re.escape(kw_lower) + r"\b"

        if re.search(pattern, text):
            found.append(kw)

    return "; ".join(sorted(set(found)))


# ============================================================
# 3. LOAD DATABASE
# ============================================================

conn = sqlite3.connect(DB_PATH)

# Check available tables
tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn
)

print("\nAvailable tables:")
print(tables)

# The uploaded DB contains comment_lexical_clusters
df = pd.read_sql_query("""
    SELECT
        comment_id,
        post_id,
        body,
        cleaned_text,
        cluster,
        color,
        svd_x,
        svd_y
    FROM comment_lexical_clusters
""", conn)

conn.close()

print("\nLoaded observations:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nCluster counts:")
print(df["cluster"].value_counts().sort_index())


# ============================================================
# 4. CLEAN DATA AND CREATE SICOPHANCY VARIABLES
# ============================================================

# Use cleaned_text when available, otherwise body
df["analysis_text"] = df["cleaned_text"]

missing_cleaned = df["analysis_text"].isna() | (df["analysis_text"].astype(str).str.strip() == "")
df.loc[missing_cleaned, "analysis_text"] = df.loc[missing_cleaned, "body"]

df["cluster"] = df["cluster"].astype(int)

df["sycophancy_keyword_count"] = df["analysis_text"].apply(count_sycophancy_keywords)

df["sycophancy_matched_keywords"] = df["analysis_text"].apply(matched_sycophancy_keywords)

df["has_sycophancy_keyword"] = (
    df["sycophancy_keyword_count"] > 0
).astype(int)

df["sycophancy_2plus"] = (
    df["sycophancy_keyword_count"] >= 2
).astype(int)

df["sycophancy_intensity"] = pd.cut(
    df["sycophancy_keyword_count"],
    bins=[-1, 0, 1, np.inf],
    labels=["none", "one_keyword", "two_or_more_keywords"]
)

print("\nSycophancy keyword distribution:")
print(df["sycophancy_keyword_count"].describe())

print("\nAny sycophancy keyword:")
print(df["has_sycophancy_keyword"].value_counts(normalize=True).mul(100).round(2))


# ============================================================
# 5. CLUSTER-LEVEL DESCRIPTIVE TABLE
# ============================================================

cluster_summary = (
    df
    .groupby("cluster")
    .agg(
        n_comments=("comment_id", "count"),
        mean_count=("sycophancy_keyword_count", "mean"),
        median_count=("sycophancy_keyword_count", "median"),
        q75_count=("sycophancy_keyword_count", lambda x: x.quantile(0.75)),
        q90_count=("sycophancy_keyword_count", lambda x: x.quantile(0.90)),
        max_count=("sycophancy_keyword_count", "max"),
        pct_any_sycophancy=("has_sycophancy_keyword", lambda x: 100 * x.mean()),
        pct_2plus_keywords=("sycophancy_2plus", lambda x: 100 * x.mean()),
        avg_svd_x=("svd_x", "mean"),
        avg_svd_y=("svd_y", "mean")
    )
    .reset_index()
    .sort_values("pct_any_sycophancy", ascending=False)
)

print("\nCluster-level sycophancy summary:")
print(cluster_summary.round(3))

cluster_summary.to_csv(
    OUTPUT_DIR / "cluster_sycophancy_summary.csv",
    index=False
)


# ============================================================
# 6. TOP MATCHED KEYWORDS BY CLUSTER
# ============================================================

keyword_rows = []

for cluster_id, group in df.groupby("cluster"):
    all_keywords = []

    for item in group["sycophancy_matched_keywords"].dropna():
        if item.strip() == "":
            continue
        all_keywords.extend([x.strip() for x in item.split(";") if x.strip()])

    if len(all_keywords) == 0:
        continue

    keyword_counts = pd.Series(all_keywords).value_counts()

    for kw, n in keyword_counts.items():
        keyword_rows.append({
            "cluster": cluster_id,
            "keyword": kw,
            "keyword_count": int(n),
            "keyword_share_within_cluster_comments": 100 * n / len(group)
        })

keyword_cluster_df = pd.DataFrame(keyword_rows)

if not keyword_cluster_df.empty:
    keyword_cluster_df = keyword_cluster_df.sort_values(
        ["cluster", "keyword_count"],
        ascending=[True, False]
    )

    print("\nTop sycophancy keywords by cluster:")
    print(keyword_cluster_df.groupby("cluster").head(10))

    keyword_cluster_df.to_csv(
        OUTPUT_DIR / "cluster_sycophancy_keywords_long.csv",
        index=False
    )


# ============================================================
# 7. KRUSKAL-WALLIS TEST ACROSS CLUSTERS
# ============================================================

groups = [
    group["sycophancy_keyword_count"].values
    for _, group in df.groupby("cluster")
]

kw = kruskal(*groups)

n = len(df)
k = df["cluster"].nunique()

epsilon_squared = (kw.statistic - k + 1) / (n - k)
epsilon_squared = max(0, epsilon_squared)

print("\nKruskal-Wallis test: cluster x sycophancy_keyword_count")
print("H statistic:", round(kw.statistic, 3))
print("p-value:", kw.pvalue)
print("epsilon-squared:", round(epsilon_squared, 4))

kw_results = pd.DataFrame([{
    "test": "Kruskal-Wallis",
    "group_var": "cluster",
    "outcome": "sycophancy_keyword_count",
    "H_statistic": kw.statistic,
    "p_value": kw.pvalue,
    "epsilon_squared": epsilon_squared,
    "n": n,
    "k_groups": k
}])

kw_results.to_csv(
    OUTPUT_DIR / "kruskal_cluster_sycophancy_results.csv",
    index=False
)


# ============================================================
# 8. PAIRWISE MANN-WHITNEY U TESTS WITH FDR CORRECTION
# ============================================================

clusters = sorted(df["cluster"].dropna().unique())

pairwise_results = []

for c1, c2 in combinations(clusters, 2):
    x = df.loc[df["cluster"] == c1, "sycophancy_keyword_count"]
    y = df.loc[df["cluster"] == c2, "sycophancy_keyword_count"]

    u_stat, p_val = mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    )

    pairwise_results.append({
        "cluster_1": c1,
        "cluster_2": c2,
        "n_1": len(x),
        "n_2": len(y),
        "mean_1": x.mean(),
        "mean_2": y.mean(),
        "median_1": x.median(),
        "median_2": y.median(),
        "pct_any_1": 100 * (x > 0).mean(),
        "pct_any_2": 100 * (y > 0).mean(),
        "u_stat": u_stat,
        "p_value_raw": p_val
    })

pairwise_df = pd.DataFrame(pairwise_results)

reject, p_adj, _, _ = multipletests(
    pairwise_df["p_value_raw"],
    method="fdr_bh"
)

pairwise_df["p_value_fdr"] = p_adj
pairwise_df["reject_fdr_5pct"] = reject

pairwise_df = pairwise_df.sort_values("p_value_fdr")

print("\nPairwise Mann-Whitney tests across clusters:")
print(pairwise_df.round(4))

pairwise_df.to_csv(
    OUTPUT_DIR / "pairwise_cluster_sycophancy_mannwhitney_fdr.csv",
    index=False
)


# ============================================================
# 9. CHI-SQUARE TEST: CLUSTER x SICOPHANCY INTENSITY
# ============================================================

intensity_table = pd.crosstab(
    df["cluster"],
    df["sycophancy_intensity"]
)

chi2, p_chi, dof, expected = chi2_contingency(intensity_table)

cramers_v = np.sqrt(
    chi2 / (
        intensity_table.values.sum()
        * min(intensity_table.shape[0] - 1, intensity_table.shape[1] - 1)
    )
)

print("\nCluster x sycophancy intensity table:")
print(intensity_table)

print("\nChi-square test: cluster x sycophancy intensity")
print("Chi-square:", round(chi2, 3))
print("Degrees of freedom:", dof)
print("p-value:", p_chi)
print("Cramer's V:", round(cramers_v, 3))

intensity_table.to_csv(
    OUTPUT_DIR / "cluster_sycophancy_intensity_table.csv"
)

chi_results = pd.DataFrame([{
    "test": "Chi-square",
    "row_var": "cluster",
    "col_var": "sycophancy_intensity",
    "chi2": chi2,
    "dof": dof,
    "p_value": p_chi,
    "cramers_v": cramers_v
}])

chi_results.to_csv(
    OUTPUT_DIR / "chi_square_cluster_sycophancy_intensity_results.csv",
    index=False
)


# ============================================================
# 10. STANDARDIZED RESIDUALS FOR CHI-SQUARE
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
    OUTPUT_DIR / "cluster_sycophancy_intensity_standardized_residuals.csv"
)


# ============================================================
# 11. PERMUTATION TEST
# ============================================================

def between_group_rank_variance(data, y_col, group_col):
    temp = data[[y_col, group_col]].copy()
    temp["rank_y"] = temp[y_col].rank(method="average")

    group_means = temp.groupby(group_col)["rank_y"].mean()
    group_sizes = temp.groupby(group_col)["rank_y"].size()
    grand_mean = temp["rank_y"].mean()

    return np.average(
        (group_means - grand_mean) ** 2,
        weights=group_sizes
    )


observed_stat = between_group_rank_variance(
    df,
    y_col="sycophancy_keyword_count",
    group_col="cluster"
)

rng = np.random.default_rng(123)
n_perm = 1000
perm_stats = []

for _ in range(n_perm):
    temp = df.copy()
    temp["cluster_perm"] = rng.permutation(temp["cluster"].values)

    perm_stats.append(
        between_group_rank_variance(
            temp,
            y_col="sycophancy_keyword_count",
            group_col="cluster_perm"
        )
    )

perm_stats = np.array(perm_stats)

perm_p = (np.sum(perm_stats >= observed_stat) + 1) / (n_perm + 1)

print("\nPermutation test on between-cluster rank variance")
print("Observed statistic:", observed_stat)
print("Permutation p-value:", perm_p)

perm_results = pd.DataFrame([{
    "test": "Permutation test",
    "group_var": "cluster",
    "outcome": "sycophancy_keyword_count",
    "observed_statistic": observed_stat,
    "n_permutations": n_perm,
    "p_value": perm_p
}])

perm_results.to_csv(
    OUTPUT_DIR / "permutation_cluster_sycophancy_results.csv",
    index=False
)


# ============================================================
# 12. SAVE FINAL DATASET
# ============================================================

df.to_csv(
    OUTPUT_DIR / "comment_cluster_sycophancy_dataset.csv",
    index=False
)

print("\nSaved files:")
print(OUTPUT_DIR / "cluster_sycophancy_summary.csv")
print(OUTPUT_DIR / "cluster_sycophancy_keywords_long.csv")
print(OUTPUT_DIR / "kruskal_cluster_sycophancy_results.csv")
print(OUTPUT_DIR / "pairwise_cluster_sycophancy_mannwhitney_fdr.csv")
print(OUTPUT_DIR / "cluster_sycophancy_intensity_table.csv")
print(OUTPUT_DIR / "chi_square_cluster_sycophancy_intensity_results.csv")
print(OUTPUT_DIR / "cluster_sycophancy_intensity_standardized_residuals.csv")
print(OUTPUT_DIR / "permutation_cluster_sycophancy_results.csv")
print(OUTPUT_DIR / "comment_cluster_sycophancy_dataset.csv")