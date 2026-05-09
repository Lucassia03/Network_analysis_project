import sqlite3
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

# Backend non interattivo per Linux/server senza interfaccia grafica
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from sentence_transformers import SentenceTransformer
from networkx.algorithms.community import greedy_modularity_communities


# ==========================================================
# 1. Settings
# ==========================================================

db_path = "moltbook_final_NLI_sim.db"

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

output_png = "semantic_graph_posts_only.png"
output_graphml = "semantic_graph_posts_only.graphml"
output_nodes_csv = "semantic_graph_posts_nodes.csv"
output_edges_csv = "semantic_graph_posts_edges.csv"

batch_size = 64
post_threshold = 0.45

# Se il modello è già in cache sul server, lascia True.
# Se dà errore perché non trova il modello, metti False solo per scaricarlo.
local_files_only = True


# ==========================================================
# 2. Load posts from SQLite
# ==========================================================

print("Loading posts from database...")

conn = sqlite3.connect(db_path)

# Controlliamo se la tabella posts ha anche title
columns = pd.read_sql("PRAGMA table_info(posts);", conn)["name"].tolist()

if "title" in columns:
    posts_df = pd.read_sql(
        """
        SELECT id, title, body
        FROM posts
        WHERE body IS NOT NULL
          AND TRIM(body) != '';
        """,
        conn
    )

    posts_df["title"] = posts_df["title"].fillna("").astype(str)
    posts_df["body"] = posts_df["body"].fillna("").astype(str)

    posts_df["post_text"] = (
        posts_df["title"].str.strip() + "\n" + posts_df["body"].str.strip()
    ).str.strip().str.lower()

else:
    posts_df = pd.read_sql(
        """
        SELECT id, body
        FROM posts
        WHERE body IS NOT NULL
          AND TRIM(body) != '';
        """,
        conn
    )

    posts_df["body"] = posts_df["body"].fillna("").astype(str)
    posts_df["post_text"] = posts_df["body"].str.strip().str.lower()

conn.close()

posts_df = posts_df[
    posts_df["post_text"].str.strip() != ""
].copy()

posts_df["post_node"] = "post_" + posts_df["id"].astype(str)

print("Posts loaded:", len(posts_df))

if len(posts_df) == 0:
    raise ValueError("No valid posts found.")


# ==========================================================
# 3. Load semantic model
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
# 4. Encode posts
# ==========================================================

print("Encoding posts...")

post_ids = posts_df["post_node"].tolist()
post_texts = posts_df["post_text"].tolist()

post_embeddings = model.encode(
    post_texts,
    batch_size=batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ==========================================================
# 5. Compute post-post similarity
# ==========================================================
# Embeddings normalized: cosine similarity = dot product

print("Computing post-post similarities...")

post_post_sim = post_embeddings @ post_embeddings.T


# ==========================================================
# 6. Build graph with only posts
# ==========================================================

print("Building post graph...")

G = nx.Graph()

for _, row in posts_df.iterrows():
    G.add_node(
        row["post_node"],
        kind="post",
        post_id=int(row["id"]),
        text=str(row["post_text"])[:1000]
    )

edge_rows = []

for i in range(len(post_ids)):
    for j in range(i + 1, len(post_ids)):
        sim = float(post_post_sim[i, j])

        if sim >= post_threshold:
            source = post_ids[i]
            target = post_ids[j]

            G.add_edge(
                source,
                target,
                kind="post_post_similarity",
                weight=sim,
                similarity=sim
            )

            edge_rows.append({
                "source": source,
                "target": target,
                "kind": "post_post_similarity",
                "weight": sim,
                "similarity": sim
            })

print("Graph built.")
print("Posts/nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# ==========================================================
# 7. Communities
# ==========================================================

print("Detecting communities...")

if G.number_of_edges() > 0:
    communities = list(
        greedy_modularity_communities(
            G,
            weight="weight"
        )
    )
else:
    communities = [{node} for node in G.nodes()]

cluster_map = {}

for cluster_id, community in enumerate(communities):
    for node in community:
        cluster_map[node] = cluster_id

nx.set_node_attributes(G, cluster_map, "cluster")

print("Communities detected:", len(communities))


# ==========================================================
# 8. Summary
# ==========================================================

print("\nPOST GRAPH SUMMARY")
print("Posts:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("\nCLUSTERS")

cluster_sizes = {}

for cid in set(cluster_map.values()):
    cluster_sizes[cid] = sum(
        1 for node in G.nodes()
        if cluster_map[node] == cid
    )

for cid, size in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True):
    print(f"cluster {cid}: {size} posts")


# ==========================================================
# 9. Save nodes and edges CSV
# ==========================================================

print("Saving CSV files...")

nodes_records = []

for node, attrs in G.nodes(data=True):
    record = {"node": node}
    record.update(attrs)
    nodes_records.append(record)

nodes_df = pd.DataFrame(nodes_records)
nodes_df.to_csv(output_nodes_csv, index=False)

edges_df = pd.DataFrame(edge_rows)
edges_df.to_csv(output_edges_csv, index=False)


# ==========================================================
# 10. Save GraphML
# ==========================================================

print("Saving GraphML...")

nx.write_graphml(G, output_graphml)


# ==========================================================
# 11. Visualization
# ==========================================================
# Qui il grafo è piccolo, circa 675 post, quindi spring_layout va bene.

print("Computing layout and saving PNG...")

plt.figure(figsize=(18, 14), dpi=180)

pos = nx.spring_layout(
    G,
    seed=42,
    k=0.35,
    iterations=80,
    weight="weight"
)

post_nodes = list(G.nodes())

pp_edges = [
    (u, v)
    for u, v, d in G.edges(data=True)
    if d["kind"] == "post_post_similarity"
]

post_colors = [
    cluster_map[node]
    for node in post_nodes
]

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=pp_edges,
    width=1.0,
    alpha=0.20
)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=post_nodes,
    node_size=180,
    node_color=post_colors,
    cmap=plt.cm.tab20,
    edgecolors="black",
    linewidths=0.5,
    alpha=0.95
)

plt.title("Semantic Graph of Posts", fontsize=18, pad=20)
plt.axis("off")
plt.tight_layout()

plt.savefig(
    output_png,
    bbox_inches="tight"
)

plt.close()

print("Done.")
print(f"Saved PNG: {output_png}")
print(f"Saved GraphML: {output_graphml}")
print(f"Saved nodes CSV: {output_nodes_csv}")
print(f"Saved edges CSV: {output_edges_csv}")