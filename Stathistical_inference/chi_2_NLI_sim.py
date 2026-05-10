import pandas as pd
import sqlite3
from scipy.stats import chi2_contingency

conn = sqlite3.connect("moltbook_final_NLI_sim.db")

df = pd.read_sql("""
SELECT prediction, child_depth
FROM parent_child_similarity
""", conn)

# Raggruppa depth
df["depth_group"] = pd.cut(
    df["child_depth"],
    bins=[0,1,2,100],
    labels=["1","2","3+"]
)

# Tabella contingenza
table = pd.crosstab(df["depth_group"], df["prediction"])

print(table)

chi2, p, dof, expected = chi2_contingency(table)

print("Chi2:", chi2)
print("p-value:", p)