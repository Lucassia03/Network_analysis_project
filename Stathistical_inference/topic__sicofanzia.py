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

nmf_db_path = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\DATASET GAB\moltbook_final_NLI_sim_WITH_CLUSTER_AGENT_MAPPING 2 (1).db"
)

syc_db_path = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\DATASET GAB\moltbook_merged_4_dbs_all_tables.db"
)

output_dir = Path(
    r"C:\Users\Gaetano\OneDrive\Documenti\Python VS Code\ANALISI FINALE\DATASET GAB"
)

print("NMF DB exists:", nmf_db_path.exists())
print("Sycophancy DB exists:", syc_db_path.exists())


# ============================================================
# 2. LOAD NMF TOPICS
# ============================================================

conn_nmf = sqlite3.connect(nmf_db_path)

nmf = pd.read_sql_query("""
    SELECT
        comment_id,
        author,
        body,
        dominant_topic,
        topic_weight,
        topic_confidence,
        depth
    FROM nmf_all_comment_topics
""", conn_nmf)

topic_terms = pd.read_sql_query("""
    SELECT
        topic,
        term_rank,
        term,
        weight
    FROM nmf_all_topic_terms
    WHERE term_rank <= 8
    ORDER BY topic, term_rank
""", conn_nmf)

conn_nmf.close()

topic_label_map = (
    topic_terms
    .groupby("topic")["term"]
    .apply(lambda x: ", ".join(x.astype(str).tolist()))
    .to_dict()
)

nmf["topic_terms"] = nmf["dominant_topic"].map(topic_label_map)


# ============================================================
# 3. LOAD SYCOPHANCY DATA
# ============================================================

conn_syc = sqlite3.connect(syc_db_path)

syc = pd.read_sql_query("""
    SELECT
        comment_id,
        sycophancy_keyword_count,
        has_sycophancy_keyword
    FROM comment_clusters
""", conn_syc)

conn_syc.close()


# ============================================================
# 4. MERGE
# ============================================================

nmf["comment_id"] = nmf["comment_id"].astype(str)
syc["comment_id"] = syc["comment_id"].astype(str)

df = nmf.merge(syc, on="comment_id", how="inner")

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

print("Merged observations:", len(df))
print("Number of dominant topics:", df["dominant_topic"].nunique())


# ============================================================
# 5. DESCRIPTIVE TOPIC TABLE
# ============================================================

topic_summary = (
    df
    .groupby("dominant_topic")
    .agg(
        n_comments=("comment_id", "count"),
        topic_terms=("topic_terms", "first"),
        mean_count=("sycophancy_keyword_count", "mean"),
        median_count=("sycophancy_keyword_count", "median"),
        q75_count=("sycophancy_keyword_count", lambda x: x.quantile(0.75)),
        q90_count=("sycophancy_keyword_count", lambda x: x.quantile(0.90)),
        pct_any_sycophancy=("has_sycophancy_keyword", lambda x: 100 * x.mean()),
        pct_2plus_keywords=("sycophancy_2plus", lambda x: 100 * x.mean())
    )
    .reset_index()
    .sort_values("pct_any_sycophancy", ascending=False)
)

print("\nTopic-level sycophancy summary:")
print(topic_summary.round(3))

topic_summary.to_csv(
    output_dir / "nonparametric_topic_sycophancy_summary.csv",
    index=False
)


# ============================================================
# 6. KRUSKAL-WALLIS TEST
# ============================================================

groups = [
    group["sycophancy_keyword_count"].values
    for _, group in df.groupby("dominant_topic")
]

kw = kruskal(*groups)

n = len(df)
k = df["dominant_topic"].nunique()

epsilon_squared = (kw.statistic - k + 1) / (n - k)

print("\nKruskal-Wallis test")
print("H statistic:", round(kw.statistic, 3))
print("p-value:", kw.pvalue)
print("epsilon-squared:", round(epsilon_squared, 4))


# ============================================================
# 7. PAIRWISE MANN-WHITNEY U TESTS WITH FDR CORRECTION
# ============================================================

topics = sorted(df["dominant_topic"].dropna().unique())

pairwise_results = []

for t1, t2 in combinations(topics, 2):
    x = df.loc[df["dominant_topic"] == t1, "sycophancy_keyword_count"]
    y = df.loc[df["dominant_topic"] == t2, "sycophancy_keyword_count"]
    
    u_stat, p_val = mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    )
    
    pairwise_results.append({
        "topic_1": t1,
        "topic_2": t2,
        "n_1": len(x),
        "n_2": len(y),
        "median_1": x.median(),
        "median_2": y.median(),
        "mean_1": x.mean(),
        "mean_2": y.mean(),
        "u_stat": u_stat,
        "p_value_raw": p_val
    })

pairwise_df = pd.DataFrame(pairwise_results)

# Benjamini-Hochberg FDR correction
reject, p_adj, _, _ = multipletests(
    pairwise_df["p_value_raw"],
    method="fdr_bh"
)

pairwise_df["p_value_fdr"] = p_adj
pairwise_df["reject_fdr_5pct"] = reject

# Add topic terms
pairwise_df["topic_1_terms"] = pairwise_df["topic_1"].map(topic_label_map)
pairwise_df["topic_2_terms"] = pairwise_df["topic_2"].map(topic_label_map)

pairwise_df = pairwise_df.sort_values("p_value_fdr")

print("\nPairwise Mann-Whitney results:")
print(pairwise_df.head(30))

pairwise_df.to_csv(
    output_dir / "nonparametric_pairwise_topic_sycophancy_mannwhitney_fdr.csv",
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
    df["dominant_topic"],
    df["sycophancy_intensity"]
)

chi2, p_chi, dof, expected = chi2_contingency(intensity_table)

cramers_v = np.sqrt(
    chi2 / (intensity_table.values.sum() * min(intensity_table.shape[0] - 1, intensity_table.shape[1] - 1))
)

print("\nChi-square test: topic x sycophancy intensity category")
print("Chi-square:", round(chi2, 3))
print("Degrees of freedom:", dof)
print("p-value:", p_chi)
print("Cramer's V:", round(cramers_v, 3))

intensity_table.to_csv(
    output_dir / "nonparametric_topic_sycophancy_intensity_table.csv"
)


# ============================================================
# 9. OPTIONAL: PERMUTATION TEST
# ============================================================

def between_topic_rank_variance(data, y_col, topic_col):
    temp = data[[y_col, topic_col]].copy()
    temp["rank_y"] = temp[y_col].rank(method="average")
    
    topic_means = temp.groupby(topic_col)["rank_y"].mean()
    topic_sizes = temp.groupby(topic_col)["rank_y"].size()
    grand_mean = temp["rank_y"].mean()
    
    return np.average(
        (topic_means - grand_mean) ** 2,
        weights=topic_sizes
    )

observed_stat = between_topic_rank_variance(
    df,
    y_col="sycophancy_keyword_count",
    topic_col="dominant_topic"
)

rng = np.random.default_rng(123)
n_perm = 1000
perm_stats = []

for _ in range(n_perm):
    temp = df.copy()
    temp["dominant_topic_perm"] = rng.permutation(temp["dominant_topic"].values)
    
    perm_stats.append(
        between_topic_rank_variance(
            temp,
            y_col="sycophancy_keyword_count",
            topic_col="dominant_topic_perm"
        )
    )

perm_stats = np.array(perm_stats)

perm_p = (np.sum(perm_stats >= observed_stat) + 1) / (n_perm + 1)

print("\nPermutation test on between-topic rank variance")
print("Observed statistic:", observed_stat)
print("Permutation p-value:", perm_p)