import sqlite3
import pandas as pd
import networkx as nx


# ==========================================================
# 1. Settings
# ==========================================================

db_path = "moltbook_final_NLI_sim.db"

output_graphml = "bipartite_posts_comments.graphml"
output_nodes_csv = "bipartite_posts_comments_nodes.csv"
output_edges_csv = "bipartite_posts_comments_edges.csv"


# ==========================================================
# 2. Load data
# ==========================================================

print("Loading database...")

conn = sqlite3.connect(db_path)

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)["name"].tolist()

if "comment_post_nli" in tables:
    print("Using table: comment_post_nli")

    df = pd.read_sql(
        """
        SELECT
            post_id,
            comment_id,
            post_author,
            comment_author,
            parent_comment_id,
            comment_depth,
            post_text,
            comment_text,
            contradiction_score,
            neutral_score,
            entailment_score,
            prediction,
            similarity
        FROM comment_post_nli
        WHERE post_text IS NOT NULL
          AND comment_text IS NOT NULL
          AND TRIM(post_text) != ''
          AND TRIM(comment_text) != '';
        """,
        conn
    )

else:
    print("Table comment_post_nli not found. Falling back to posts/comments.")

    posts = pd.read_sql(
        """
        SELECT
            id AS post_id,
            author AS post_author,
            body AS post_text
        FROM posts
        WHERE body IS NOT NULL
          AND TRIM(body) != '';
        """,
        conn
    )

    comments = pd.read_sql(
        """
        SELECT
            id AS comment_id,
            post_id,
            author AS comment_author,
            parent_comment_id,
            depth AS comment_depth,
            body AS comment_text
        FROM comments
        WHERE body IS NOT NULL
          AND TRIM(body) != '';
        """,
        conn
    )

    df = comments.merge(posts, on="post_id", how="inner")

    df["contradiction_score"] = ""
    df["neutral_score"] = ""
    df["entailment_score"] = ""
    df["prediction"] = ""
    df["similarity"] = ""

conn.close()

print("Rows loaded:", len(df))


# ==========================================================
# 3. Prepare IDs
# ==========================================================

df["post_node"] = "post_" + df["post_id"].astype(str)
df["comment_node"] = "comment_" + df["comment_id"].astype(str)

posts_df = (
    df[["post_id", "post_node", "post_author", "post_text"]]
    .drop_duplicates("post_id")
    .reset_index(drop=True)
)

comments_df = (
    df[
        [
            "comment_id",
            "comment_node",
            "post_id",
            "post_node",
            "comment_author",
            "parent_comment_id",
            "comment_depth",
            "comment_text",
            "contradiction_score",
            "neutral_score",
            "entailment_score",
            "prediction",
            "similarity"
        ]
    ]
    .drop_duplicates("comment_id")
    .reset_index(drop=True)
)

print("Unique posts:", len(posts_df))
print("Unique comments:", len(comments_df))


# ==========================================================
# 4. Build bipartite graph
# ==========================================================

print("Building bipartite graph...")

G = nx.Graph()

# Add post nodes
for _, row in posts_df.iterrows():
    G.add_node(
        row["post_node"],
        kind="post",
        node_type="post",
        post_id=int(row["post_id"]),
        author=str(row["post_author"]),
        text=str(row["post_text"])[:500]
    )

# Add comment nodes
for _, row in comments_df.iterrows():
    G.add_node(
        row["comment_node"],
        kind="comment",
        node_type="comment",
        comment_id=int(row["comment_id"]),
        post_id=int(row["post_id"]),
        parent_post=row["post_node"],
        author=str(row["comment_author"]),
        parent_comment_id="" if pd.isna(row["parent_comment_id"]) else str(row["parent_comment_id"]),
        depth="" if pd.isna(row["comment_depth"]) else int(row["comment_depth"]),
        prediction=str(row["prediction"]),
        contradiction_score="" if pd.isna(row["contradiction_score"]) else float(row["contradiction_score"]),
        neutral_score="" if pd.isna(row["neutral_score"]) else float(row["neutral_score"]),
        entailment_score="" if pd.isna(row["entailment_score"]) else float(row["entailment_score"]),
        similarity="" if pd.isna(row["similarity"]) else float(row["similarity"]),
        text=str(row["comment_text"])[:500]
    )

# Add only post-comment edges
for _, row in comments_df.iterrows():
    sim = row["similarity"]

    if pd.isna(sim):
        weight = 1.0
        similarity = ""
    else:
        weight = float(sim)
        similarity = float(sim)

    G.add_edge(
        row["post_node"],
        row["comment_node"],
        kind="post_comment",
        edge_type="post_comment",
        weight=weight,
        similarity=similarity,
        prediction=str(row["prediction"])
    )

print("Graph built.")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("Expected nodes:", len(posts_df) + len(comments_df))
print("Expected edges:", len(comments_df))


# ==========================================================
# 5. Add component id
# ==========================================================
# Ogni componente dovrebbe corrispondere a un post con i suoi commenti.

component_map = {}

for component_id, component in enumerate(nx.connected_components(G)):
    for node in component:
        component_map[node] = component_id

nx.set_node_attributes(G, component_map, "component")

print("Connected components:", len(set(component_map.values())))


# ==========================================================
# 6. Save CSV files
# ==========================================================

print("Saving nodes CSV...")

nodes_records = []

for node, attrs in G.nodes(data=True):
    record = {"node": node}
    record.update(attrs)
    nodes_records.append(record)

nodes_out = pd.DataFrame(nodes_records)
nodes_out.to_csv(output_nodes_csv, index=False)


print("Saving edges CSV...")

edges_records = []

for source, target, attrs in G.edges(data=True):
    record = {
        "source": source,
        "target": target
    }
    record.update(attrs)
    edges_records.append(record)

edges_out = pd.DataFrame(edges_records)
edges_out.to_csv(output_edges_csv, index=False)


# ==========================================================
# 7. Save GraphML
# ==========================================================

print("Saving GraphML...")

nx.write_graphml(G, output_graphml)

print("Done.")
print(f"Saved GraphML: {output_graphml}")
print(f"Saved nodes CSV: {output_nodes_csv}")
print(f"Saved edges CSV: {output_edges_csv}")