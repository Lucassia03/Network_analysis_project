# ============================================
# LOGISTIC REGRESSION 1
# Entailment ~ Child Depth
# ============================================

import pandas as pd
import sqlite3
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt

# Connessione database
conn = sqlite3.connect("moltbook_final_NLI_sim.db")

# Import dati
df = pd.read_sql("""
SELECT prediction,
       child_depth
FROM parent_child_similarity
""", conn)

# Variabile dipendente
df["entailment"] = (
    df["prediction"] == "entailment"
).astype(int)

# Child depth categorica
df["depth_group"] = pd.cut(
    df["child_depth"],
    bins=[0,1,2,100],
    labels=["1","2","3+"]
)

# Modello logit
model_1 = smf.logit(
    "entailment ~ C(depth_group)",
    data=df
).fit()

# Risultati
print(model_1.summary())

# Odds ratios
print("\nOdds Ratios:")
print(np.exp(model_1.params))

# ============================================
# Grafico probabilità predette
# ============================================

pred_df = pd.DataFrame({
    "depth_group": ["1", "2", "3+"]
})

pred_df["predicted_prob"] = model_1.predict(pred_df)

plt.figure(figsize=(7,5))

plt.bar(
    pred_df["depth_group"],
    pred_df["predicted_prob"]
)

plt.xlabel("Child Depth")
plt.ylabel("Predicted Probability of Entailment")
plt.title("Predicted Probability of Entailment by Thread Depth")

plt.show()

# ============================================
# LOGISTIC REGRESSION 2
# Entailment ~ Similarity + Child Depth
# ============================================

# ============================================
# LOGISTIC REGRESSION
# Entailment ~ Similarity + Child Depth
# ============================================

import pandas as pd
import sqlite3
import statsmodels.formula.api as smf
import numpy as np

# Connessione database
conn = sqlite3.connect("moltbook_final_NLI_sim.db")

# Import dati
df = pd.read_sql("""
SELECT prediction,
       similarity,
       child_depth
FROM parent_child_similarity
""", conn)

# Variabile dipendente:
# 1 = entailment
# 0 = neutral o contradiction
df["entailment"] = (
    df["prediction"] == "entailment"
).astype(int)

# Child depth categorica
df["depth_group"] = pd.cut(
    df["child_depth"],
    bins=[0,1,2,100],
    labels=["1","2","3+"]
)

# Regressione logistica
# similarity = continua
# depth_group = categorica
model_2 = smf.logit(
    "entailment ~ similarity + C(depth_group)",
    data=df
).fit()

# Risultati modello
print(model_2.summary())

# Odds ratios
print("\nOdds Ratios:")
print(np.exp(model_2.params))