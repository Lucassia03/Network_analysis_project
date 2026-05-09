import sqlite3
import numpy as np
import pandas as pd
import networkx as nx
import torch

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ==========================================================
# 1. Settings
# ==========================================================

db_path = "moltbook_final_NLI_sim.db"

output_graphml = "semantic_graph_posts_comments.graphml"
output_nodes_csv = "semantic_graph_nodes.csv"
output_edges_csv = "semantic_graph_edges.csv"

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

batch_size = 128

# Soglia similarity post-commento.
# Se nel database hai già la colonna similarity, useremo quella.
post_comment_threshold = 0.40

# Soglia similarity commento-commento dentro lo stesso post.
comment_comment_threshold = 0.55

# Limita gli edge comment-comment per evitare grafi ingestibili.
# Per ogni commento teniamo massimo top_k_comment_edges collegamenti simili.
top_k_comment_edges = 5

# Se vuoi testare su pochi post, metti ad esempio 100.
# Per usare tutto, lascia None.
max_posts = None

# Importante su nodi GPU senza internet.
# Se il modello non è in cache, metti False solo per scaricarlo una volta.
local_files_only = True


# ==========================================================
# 2. Load data from database
# ==========================================================

print("Loading database...")

conn = sqlite3.connect(db_path)

# Usiamo la tabella già arricchita con NLI e similarity, se esiste.
tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)["name"].tolist()

if "comment_post_nli" in tables:
    print("Using table: comment_post_nli")

    if max_posts is None:
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
              AND TRIM(comment_text) != ''
              AND post_id IN (
                  SELECT DISTINCT post_id
                  FROM comment_post_nli
                  LIMIT ?
              );
            """,
            conn,
            params=(max_posts,)
        )

else:
    print("Table comment_post_nli not found. Falling back to posts/comments.")

    if max_posts is None:
        posts = pd.read_sql(
            """
            SELECT id AS post_id, author AS post_author, body AS post_text
            FROM posts
            WHERE body IS NOT NULL
              AND TRIM(body) != '';
            """,
            conn
        )
    else:
        posts = pd.read_sql(
            """
            SELECT id AS post_id, author AS post_author, body AS post_text
            FROM posts
            WHERE body IS NOT NULL
              AND TRIM(body) != ''
            LIMIT ?;
            """,
            conn,
            params=(max_posts,)
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

    df = comments.merge(
        posts,
        on="post_id",
        how="inner"
    )

    df["contradiction_score"] = np.nan
    df["neutral_score"] = np.nan
    df["entailment_score"] = np.nan
    df["prediction"] = ""
    df["similarity"] = np.nan

conn.close()

print("Rows loaded:", len(df))

df["post_text"] = df["post_text"].fillna("").astype(str).str.strip().str.lower()
df["comment_text"] = df["comment_text"].fillna("").astype(str).str.strip().str.lower()

df = df[
    (df["post_text"] != "") &
    (df["comment_text"] != "")
].copy()

print("Rows after cleaning:", len(df))

if len(df) == 0:
    raise ValueError("No valid rows found.")


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
# 4. Load embedding model
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading embedding model...")

model = SentenceTransformer(
    embedding_model_name,
    device=device,
    local_files_only=local_files_only
)


# ==========================================================
# 5. Encode comments only
# ==========================================================
# Per il grafo comment-comment servono embeddings dei commenti.
# Per post-comment usiamo già la similarity salvata nella tabella.

print("Encoding comments...")

comment_ids = comments_df["comment_node"].tolist()
comment_texts = comments_df["comment_text"].tolist()

comment_embeddings = model.encode(
    comment_texts,
    batch_size=batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

comment_index = {
    comment_id: idx
    for idx, comment_id in enumerate(comment_ids)
}


# ==========================================================
# 6. Build graph
# ==========================================================

print("Building graph...")

G = nx.Graph()


# ------------------------------
# Add post nodes
# ------------------------------

for _, row in posts_df.iterrows():
    G.add_node(
        row["post_node"],
        kind="post",
        post_id=int(row["post_id"]),
        author=str(row["post_author"]),
        text=str(row["post_text"])[:500]
    )


# ------------------------------
# Add comment nodes
# ------------------------------

for _, row in comments_df.iterrows():
    G.add_node(
        row["comment_node"],
        kind="comment",
        comment_id=int(row["comment_id"]),
        post_id=int(row["post_id"]),
        parent_post=row["post_node"],
        author=str(row["comment_author"]),
        parent_comment_id="" if pd.isna(row["parent_comment_id"]) else str(row["parent_comment_id"]),
        depth="" if pd.isna(row["comment_depth"]) else int(row["comment_depth"]),
        prediction=str(row["prediction"]),
        contradiction_score=float(row["contradiction_score"]) if pd.notna(row["contradiction_score"]) else np.nan,
        neutral_score=float(row["neutral_score"]) if pd.notna(row["neutral_score"]) else np.nan,
        entailment_score=float(row["entailment_score"]) if pd.notna(row["entailment_score"]) else np.nan,
        post_comment_similarity=float(row["similarity"]) if pd.notna(row["similarity"]) else np.nan,
        text=str(row["comment_text"])[:500]
    )


# ==========================================================
# 7. Add post-comment edges
# ==========================================================

print("Adding post-comment edges...")

for _, row in tqdm(comments_df.iterrows(), total=len(comments_df)):
    sim = row["similarity"]

    if pd.isna(sim):
        edge_kind = "ownership"
        edge_weight = 0.15
        sim_value = np.nan
    else:
        sim_value = float(sim)

        if sim_value >= post_comment_threshold:
            edge_kind = "post_comment_similarity"
            edge_weight = sim_value
        else:
            edge_kind = "ownership"
            edge_weight = 0.15

    G.add_edge(
        row["post_node"],
        row["comment_node"],
        kind=edge_kind,
        weight=float(edge_weight),
        similarity=sim_value
    )


# ==========================================================
# 8. Add comment-comment edges inside each post
# ==========================================================

print("Adding comment-comment edges inside each post...")

edge_rows = []

for post_id, group in tqdm(comments_df.groupby("post_id")):
    group_comment_nodes = group["comment_node"].tolist()

    if len(group_comment_nodes) < 2:
        continue

    idxs = [
        comment_index[cid]
        for cid in group_comment_nodes
    ]

    emb = comment_embeddings[idxs]

    # Embeddings normalizzati: cosine similarity = dot product
    sim_mat = emb @ emb.T

    n = len(group_comment_nodes)

    for i in range(n):
        sims = sim_mat[i].copy()
        sims[i] = -1.0

        candidate_idx = np.where(sims >= comment_comment_threshold)[0]

        if len(candidate_idx) == 0:
            continue

        # Teniamo solo i top-k più simili per evitare milioni di edge
        candidate_idx = candidate_idx[
            np.argsort(sims[candidate_idx])[::-1]
        ][:top_k_comment_edges]

        for j in candidate_idx:
            if i >= j:
                continue

            sim = float(sims[j])

            source = group_comment_nodes[i]
            target = group_comment_nodes[j]

            G.add_edge(
                source,
                target,
                kind="comment_comment_similarity",
                weight=sim,
                similarity=sim
            )


print("Graph built.")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# ==========================================================
# 9. Add simple connected-component clusters
# ==========================================================
# Molto più veloce di greedy_modularity_communities su grafi grandi.

print("Computing connected components...")

component_map = {}

for component_id, component in enumerate(nx.connected_components(G)):
    for node in component:
        component_map[node] = component_id

nx.set_node_attributes(G, component_map, "component")

print("Connected components:", len(set(component_map.values())))


# ==========================================================
# 10. Save nodes and edges as CSV
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
# 11. Save GraphML
# ==========================================================

print("Saving GraphML...")

# GraphML non gestisce bene NaN: convertiamo NaN in stringa vuota.
for node, attrs in G.nodes(data=True):
    for key, value in list(attrs.items()):
        if isinstance(value, float) and np.isnan(value):
            attrs[key] = ""

for source, target, attrs in G.edges(data=True):
    for key, value in list(attrs.items()):
        if isinstance(value, float) and np.isnan(value):
            attrs[key] = ""

nx.write_graphml(G, output_graphml)

print("Done.")
print(f"Saved GraphML: {output_graphml}")
print(f"Saved nodes CSV: {output_nodes_csv}")
print(f"Saved edges CSV: {output_edges_csv}")