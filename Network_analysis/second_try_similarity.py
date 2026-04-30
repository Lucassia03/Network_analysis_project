import sqlite3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import greedy_modularity_communities
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1) Load data from SQLite
# --------------------------------------------------
db_path = "Moltbook.db"   # change if needed

conn = sqlite3.connect(db_path)
cur = conn.cursor()

posts_rows = cur.execute("""
    SELECT id, body
    FROM posts
    WHERE body IS NOT NULL
""").fetchall()

comments_rows = cur.execute("""
    SELECT id, post_id, body
    FROM comments
    WHERE body IS NOT NULL
""").fetchall()

conn.close()


# --------------------------------------------------
# 2) Build dictionaries
# --------------------------------------------------
posts = {}
for post_id, body_post in posts_rows:
    posts[f"post_{post_id}"] = body_post

all_comments = {}
comment_to_post = {}

for comment_id, post_id, body_comment in comments_rows:
    comment_key = f"comment_{comment_id}"
    post_key = f"post_{post_id}"

    # keep only comments whose parent post exists
    if post_key in posts:
        all_comments[comment_key] = body_comment
        comment_to_post[comment_key] = post_key


# comments grouped by post
comments_by_post = {}
for comment_id, post_id in comment_to_post.items():
    comments_by_post.setdefault(post_id, {})
    comments_by_post[post_id][comment_id] = all_comments[comment_id]


# --------------------------------------------------
# 3) Semantic model
# --------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 4) Encode posts and comments
# --------------------------------------------------
post_ids = list(posts.keys())
post_texts = [posts[pid] for pid in post_ids]
post_embeddings = model.encode(post_texts, convert_to_numpy=True)

comment_ids = list(all_comments.keys())
comment_texts = [all_comments[cid] for cid in comment_ids]
comment_embeddings = model.encode(comment_texts, convert_to_numpy=True)

post_index = {pid: i for i, pid in enumerate(post_ids)}
comment_index = {cid: i for i, cid in enumerate(comment_ids)}


# --------------------------------------------------
# 5) Similarity matrices
# --------------------------------------------------
post_post_sim = cosine_similarity(post_embeddings)
comment_comment_sim = cosine_similarity(comment_embeddings)

# post -> its own comments only
post_comment_sim_by_post = {}

for post_id in post_ids:
    own_comment_ids = list(comments_by_post.get(post_id, {}).keys())

    if not own_comment_ids:
        continue

    p_emb = post_embeddings[post_index[post_id]].reshape(1, -1)
    c_embs = np.array([comment_embeddings[comment_index[cid]] for cid in own_comment_ids])

    sim_matrix = cosine_similarity(p_emb, c_embs)   # shape (1, n_comments)
    post_comment_sim_by_post[post_id] = {
        "comment_ids": own_comment_ids,
        "matrix": sim_matrix
    }


# --------------------------------------------------
# 6) Build one heterogeneous graph
# --------------------------------------------------
G = nx.Graph()

# add post nodes
for post_id, body_post in posts.items():
    G.add_node(
        post_id,
        kind="post",
        label=post_id,
        text=body_post
    )

# add comment nodes
for comment_id, body_comment in all_comments.items():
    G.add_node(
        comment_id,
        kind="comment",
        label=comment_id,
        text=body_comment,
        parent_post=comment_to_post[comment_id]
    )


# --------------------------------------------------
# 7) Add post-post similarity edges
# --------------------------------------------------
post_threshold = 0.45

for i in range(len(post_ids)):
    for j in range(i + 1, len(post_ids)):
        sim = float(post_post_sim[i, j])
        if sim >= post_threshold:
            G.add_edge(
                post_ids[i],
                post_ids[j],
                kind="post_post_similarity",
                weight=sim
            )


# --------------------------------------------------
# 8) Add comment-comment similarity edges
# --------------------------------------------------
comment_threshold = 0.45

for i in range(len(comment_ids)):
    for j in range(i + 1, len(comment_ids)):
        sim = float(comment_comment_sim[i, j])
        if sim >= comment_threshold:
            G.add_edge(
                comment_ids[i],
                comment_ids[j],
                kind="comment_comment_similarity",
                weight=sim
            )


# --------------------------------------------------
# 9) Add post-comment edges
#    only between a post and its own comments
# --------------------------------------------------
post_comment_threshold = 0.4

for post_id, data in post_comment_sim_by_post.items():
    own_comment_ids = data["comment_ids"]
    sim_values = data["matrix"][0]

    for cid, sim in zip(own_comment_ids, sim_values):
        sim = float(sim)

        if sim >= post_comment_threshold:
            G.add_edge(
                post_id,
                cid,
                kind="post_comment_similarity",
                weight=sim
            )
        else:
            G.add_edge(
                post_id,
                cid,
                kind="ownership",
                weight=0.2
            )


# --------------------------------------------------
# 10) Cluster on the whole graph
# --------------------------------------------------
if G.number_of_edges() > 0:
    communities = list(greedy_modularity_communities(G, weight="weight"))
else:
    communities = [{node} for node in G.nodes()]

cluster_map = {}
for cluster_id, community in enumerate(communities):
    for node in community:
        cluster_map[node] = cluster_id

nx.set_node_attributes(G, cluster_map, "cluster")


# --------------------------------------------------
# 11) Print matrices
# --------------------------------------------------
print("\nPOST-POST SIMILARITY MATRIX")
for i, p1 in enumerate(post_ids):
    row = []
    for j, p2 in enumerate(post_ids):
        row.append(f"{post_post_sim[i, j]:.3f}")
    print(p1, row)

print("\nCOMMENT-COMMENT SIMILARITY MATRIX")
for i, c1 in enumerate(comment_ids):
    row = []
    for j, c2 in enumerate(comment_ids):
        row.append(f"{comment_comment_sim[i, j]:.3f}")
    print(c1, row)

print("\nPOST -> OWN COMMENTS SIMILARITY")
for post_id, data in post_comment_sim_by_post.items():
    for cid, sim in zip(data["comment_ids"], data["matrix"][0]):
        print(f"{post_id} <-> {cid}: {sim:.3f}")


# --------------------------------------------------
# 12) Print cluster assignment
# --------------------------------------------------
print("\nCLUSTERS")
for node in G.nodes():
    print(
        f"{node:15s} | "
        f"type={G.nodes[node]['kind']:8s} | "
        f"cluster={cluster_map[node]}"
    )


# --------------------------------------------------
# 13) Visualize
# --------------------------------------------------
plt.figure(figsize=(24, 18))
pos = nx.spring_layout(G, seed=42, weight="weight", k=0.7)

post_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "post"]
comment_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "comment"]

pp_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "post_post_similarity"]
cc_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "comment_comment_similarity"]
pc_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] in ["post_comment_similarity", "ownership"]]

# color by cluster
post_colors = [cluster_map[n] for n in post_nodes]
comment_colors = [cluster_map[n] for n in comment_nodes]

nx.draw_networkx_nodes(
    G, pos,
    nodelist=post_nodes,
    node_shape="s",
    node_size=100,
    node_color=post_colors,
    cmap=plt.cm.Set3
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=comment_nodes,
    node_shape="o",
    node_size=25,
    node_color=comment_colors,
    cmap=plt.cm.Set3
)

nx.draw_networkx_edges(G, pos, edgelist=pp_edges, width=2.5, alpha=0.9)
nx.draw_networkx_edges(G, pos, edgelist=cc_edges, width=1.5, alpha=0.5)
nx.draw_networkx_edges(G, pos, edgelist=pc_edges, width=1.2, alpha=0.5, style="dashed")

nx.draw_networkx_labels(
    G, pos,
    labels={n: n for n in G.nodes()},
    font_size=8
)

edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=None, font_size=6)

plt.title("Moltbook Semantic Graph: Posts + Comments")
plt.axis("off")
plt.show()