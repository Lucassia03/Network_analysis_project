import sqlite3
import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================

# Database containing the parent-child cosine similarity table
SIMILARITY_DB = "moltbook_final_NLI_sim.db"

# Database containing the clustering output
CLUSTER_DB = "cluster_agent_mapping.db"

# Input tables
SIMILARITY_TABLE = "parent_child_similarity"
CLUSTER_TABLE = "comment_clusters"

# Target semantic cluster name produced by the clustering script
TARGET_CLUSTER_NAME = "great_question_skills_goals_rule_thumb"

# Output
OUTPUT_DB = "target_cluster_similarity_mapping.db"
OUTPUT_TABLE = "target_cluster_similarity_mapping"

# ==========================================================
# Utility function
# ==========================================================

def load_sql_table(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Load a full SQLite table into a pandas DataFrame.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    finally:
        conn.close()

    return df

# ==========================================================
# Main pipeline
# ==========================================================

def main():
    # ------------------------------------------------------
    # 1. Load parent-child similarity table
    # ------------------------------------------------------

    print("Loading parent-child similarity table...")

    similarity_df = load_sql_table(
        db_path=SIMILARITY_DB,
        table_name=SIMILARITY_TABLE
    )

    print("Similarity rows:", len(similarity_df))
    print("Similarity columns:", list(similarity_df.columns))

    # ------------------------------------------------------
    # 2. Load comment clusters table
    # ------------------------------------------------------

    print("\nLoading comment clusters table...")

    clusters_df = load_sql_table(
        db_path=CLUSTER_DB,
        table_name=CLUSTER_TABLE
    )

    print("Cluster rows:", len(clusters_df))
    print("Cluster columns:", list(clusters_df.columns))

    # ------------------------------------------------------
    # 3. Select the semantic target cluster
    # ------------------------------------------------------
    # This is safer than using cluster == 2, because KMeans
    # cluster IDs may change if the data or preprocessing changes.
    #
    # The target cluster is the one identified in the clustering
    # script through TARGET_KEYWORDS.

    if "is_target_cluster" in clusters_df.columns:
        target_cluster_df = clusters_df[
            clusters_df["is_target_cluster"] == 1
        ].copy()

        selection_method = "is_target_cluster == 1"

    elif "semantic_cluster_name" in clusters_df.columns:
        target_cluster_df = clusters_df[
            clusters_df["semantic_cluster_name"] == TARGET_CLUSTER_NAME
        ].copy()

        selection_method = f"semantic_cluster_name == {TARGET_CLUSTER_NAME}"

    else:
        raise KeyError(
            "Neither 'is_target_cluster' nor 'semantic_cluster_name' "
            "was found in the cluster table. Cannot identify target cluster."
        )

    print("\nTarget cluster selection method:", selection_method)
    print("Comments in target semantic cluster:", len(target_cluster_df))

    if len(target_cluster_df) == 0:
        print("\nWARNING: No comments found in the target semantic cluster.")

    # ------------------------------------------------------
    # 4. Make sure IDs are comparable
    # ------------------------------------------------------

    similarity_df["child_id"] = pd.to_numeric(
        similarity_df["child_id"],
        errors="coerce"
    )

    target_cluster_df["comment_id"] = pd.to_numeric(
        target_cluster_df["comment_id"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # 5. Join similarity with target-cluster comments
    # ------------------------------------------------------
    # child_id is the comment ID of the child in the parent-child pair.
    # comment_id is the comment ID in the clustering table.
    #
    # This keeps only the parent-child interactions where the child
    # comment belongs to the target semantic cluster.

    mapping_df = similarity_df.merge(
        target_cluster_df,
        left_on="child_id",
        right_on="comment_id",
        how="inner",
        suffixes=("_similarity", "_cluster")
    )

    print("\nMapped rows:", len(mapping_df))

    # ------------------------------------------------------
    # 6. Rename cluster-side columns for clarity
    # ------------------------------------------------------

    rename_map = {
        "body": "cluster_comment_body",
        "cleaned_text": "cluster_comment_cleaned_text",
        "cluster": "child_cluster",
        "semantic_cluster_name": "child_semantic_cluster_name",
        "is_target_cluster": "child_is_target_cluster",
    }

    mapping_df = mapping_df.rename(
        columns={
            old_name: new_name
            for old_name, new_name in rename_map.items()
            if old_name in mapping_df.columns
        }
    )

    # ------------------------------------------------------
    # 7. Reorder useful columns
    # ------------------------------------------------------

    preferred_cols = [
        # Pair metadata
        "relation_type",
        "post_id_similarity",
        "post_id_cluster",
        "parent_id",
        "child_id",
        "comment_id",

        # Authors
        "parent_author",
        "child_author",

        # Texts
        "parent_text",
        "child_text",
        "cluster_comment_body",
        "cluster_comment_cleaned_text",

        # Parent-child metadata
        "child_depth",

        # Cosine similarity
        "similarity",

        # NLI results, if available
        "contradiction_score",
        "neutral_score",
        "entailment_score",
        "prediction",

        # Cluster information
        "child_cluster",
        "child_semantic_cluster_name",
        "child_is_target_cluster",
    ]

    cols_to_keep = [
        col for col in preferred_cols
        if col in mapping_df.columns
    ]

    remaining_cols = [
        col for col in mapping_df.columns
        if col not in cols_to_keep
    ]

    mapping_df = mapping_df[cols_to_keep + remaining_cols]

    # ------------------------------------------------------
    # 8. Save mapping to SQLite
    # ------------------------------------------------------

    print(f"\nSaving mapping to SQL table '{OUTPUT_TABLE}'...")

    conn_out = sqlite3.connect(OUTPUT_DB)
    try:
        mapping_df.to_sql(
            OUTPUT_TABLE,
            conn_out,
            if_exists="replace",
            index=False
        )
    finally:
        conn_out.close()

    
    # ------------------------------------------------------
    # 10. Preview output
    # ------------------------------------------------------

    print("\nPreview:")

    preview_cols = [
        "relation_type",
        "parent_id",
        "child_id",
        "similarity",
        "prediction",
        "child_cluster",
        "child_semantic_cluster_name",
        "parent_text",
        "child_text",
    ]

    preview_cols = [
        col for col in preview_cols
        if col in mapping_df.columns
    ]

    if len(mapping_df) > 0:
        print(mapping_df[preview_cols].head(20).to_string(index=False))
    else:
        print("No mapped rows found.")

    print("\nDone.")
    print("Output DB:", OUTPUT_DB)
    print("Output table:", OUTPUT_TABLE)

if __name__ == "__main__":
    main()

#per runnarlo
#python Mapping_cos_simil_cluster_symp.py
