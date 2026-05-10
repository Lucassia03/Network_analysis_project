import sqlite3
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy import stats


# =====================================================
# Database path
# =====================================================

DB = r"C:\Users\capod\Downloads\Dataset_NLP\moltbook_final.db"


# =====================================================
# Load data
# =====================================================

conn = sqlite3.connect(DB)

query = """
SELECT
    c.comment_id,
    c.comment_author,

    c.final_sycophancy_likelihood,
    c.sycophancy_keyword_count,

    c.llm_stance_post,
    c.llm_stance_context,

    CASE 
        WHEN c.parent_comment_id IS NOT NULL THEN 1 
        ELSE 0 
    END AS had_previous_interaction,

    a.author_dominant_bert_model AS agent_model

FROM comment_master_table c
LEFT JOIN author_master_summary a
    ON c.comment_author = a.comment_author

WHERE c.final_sycophancy_likelihood IS NOT NULL
  AND c.sycophancy_keyword_count IS NOT NULL
  AND c.llm_stance_post IS NOT NULL
  AND c.llm_stance_context IS NOT NULL
  AND a.author_dominant_bert_model IS NOT NULL;
"""

df = pd.read_sql_query(query, conn)
conn.close()


# =====================================================
# Create LLM dummy variables
# =====================================================
# Reference categories:
# llm_stance_post reference = Oppose
# llm_stance_context reference = Contradiction

df["llm_post_neutral"] = (df["llm_stance_post"] == "Neutral").astype(int)
df["llm_post_support"] = (df["llm_stance_post"] == "Support").astype(int)

df["llm_context_neutral"] = (df["llm_stance_context"] == "Neutral").astype(int)
df["llm_context_entailment"] = (df["llm_stance_context"] == "Entailment").astype(int)


# =====================================================
# Basic checks
# =====================================================

print("Dataset shape:")
print(df.shape)

print("\nfinal_sycophancy_likelihood summary:")
print(df["final_sycophancy_likelihood"].describe())

print("\nsycophancy_keyword_count summary:")
print(df["sycophancy_keyword_count"].describe())

print("\nLLM stance post distribution:")
print(df["llm_stance_post"].value_counts())

print("\nLLM stance context distribution:")
print(df["llm_stance_context"].value_counts())

print("\nAgent model distribution:")
print(df["agent_model"].value_counts())

print("\nPrevious interaction distribution:")
print(df["had_previous_interaction"].value_counts())


# =====================================================
# Helper function: sklearn linear regression with p-values
# =====================================================

def fit_sklearn_linear_model(df, feature_cols, target_col, model_name):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    categorical_cols = ["agent_model"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    try:
        encoder = OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", encoder, categorical_cols)
        ],
        remainder="drop"
    )

    model = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X, y)

    X_design = pipeline.named_steps["preprocessor"].transform(X)

    if hasattr(X_design, "toarray"):
        X_design = X_design.toarray()

    X_design_with_intercept = np.column_stack([
        np.ones(X_design.shape[0]),
        X_design
    ])

    coefs = np.concatenate([
        [pipeline.named_steps["model"].intercept_],
        pipeline.named_steps["model"].coef_
    ])

    predictions = pipeline.predict(X)

    n = X_design_with_intercept.shape[0]
    p = X_design_with_intercept.shape[1]

    r2 = r2_score(y, predictions)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p))

    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)

    residuals = y - predictions
    rss = np.sum(residuals ** 2)
    df_resid = n - p

    residual_variance = rss / df_resid

    xtx_inv = np.linalg.pinv(
        X_design_with_intercept.T @ X_design_with_intercept
    )

    covariance_matrix = residual_variance * xtx_inv

    standard_errors = np.sqrt(np.diag(covariance_matrix))

    t_scores = coefs / standard_errors

    p_values = 2 * (1 - stats.t.cdf(np.abs(t_scores), df=df_resid))

    feature_names = ["Intercept"]
    feature_names += numeric_cols

    fitted_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = fitted_encoder.get_feature_names_out(categorical_cols).tolist()

    feature_names += cat_names

    results = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "std_error": standard_errors,
        "t_score": t_scores,
        "p_value": p_values
    })

    results = results.sort_values("p_value")

    metrics = pd.DataFrame({
        "model": [model_name],
        "target": [target_col],
        "model_type": ["Sklearn LinearRegression"],
        "n_observations": [n],
        "n_parameters_including_intercept": [p],
        "r2": [r2],
        "adjusted_r2": [adjusted_r2],
        "rmse": [rmse],
        "mae": [mae],
        "residual_df": [df_resid]
    })

    predictions_df = pd.DataFrame({
        "comment_id": df["comment_id"],
        "comment_author": df["comment_author"],
        "actual": y,
        "predicted": predictions,
        "residual": residuals
    })

    return {
        "pipeline": pipeline,
        "results": results,
        "metrics": metrics,
        "predictions": predictions_df
    }


# =====================================================
# Feature sets
# =====================================================

features_without_previous_interaction = [
    "llm_post_neutral",
    "llm_post_support",
    "llm_context_neutral",
    "llm_context_entailment",
    "agent_model"
]

features_with_previous_interaction = [
    "llm_post_neutral",
    "llm_post_support",
    "llm_context_neutral",
    "llm_context_entailment",
    "agent_model",
    "had_previous_interaction"
]


# =====================================================
# Regression A
# Target: final_sycophancy_likelihood
# Without previous interaction
# =====================================================

model_likelihood_1 = fit_sklearn_linear_model(
    df=df,
    feature_cols=features_without_previous_interaction,
    target_col="final_sycophancy_likelihood",
    model_name="A: likelihood, no previous interaction"
)

print("\n\n=====================================================")
print("REGRESSION A")
print("Target: final_sycophancy_likelihood")
print("Predictors: LLM Neutral, Support, Entailment + agent model")
print("=====================================================")

print("\nModel metrics:")
print(model_likelihood_1["metrics"].to_string(index=False))

print("\nRegression results:")
print(model_likelihood_1["results"].to_string(index=False))


# =====================================================
# Regression B
# Target: final_sycophancy_likelihood
# With previous interaction
# =====================================================

model_likelihood_2 = fit_sklearn_linear_model(
    df=df,
    feature_cols=features_with_previous_interaction,
    target_col="final_sycophancy_likelihood",
    model_name="B: likelihood, with previous interaction"
)

print("\n\n=====================================================")
print("REGRESSION B")
print("Target: final_sycophancy_likelihood")
print("Predictors: LLM Neutral, Support, Entailment + agent model + previous interaction")
print("=====================================================")

print("\nModel metrics:")
print(model_likelihood_2["metrics"].to_string(index=False))

print("\nRegression results:")
print(model_likelihood_2["results"].to_string(index=False))


# =====================================================
# Regression C
# Target: sycophancy_keyword_count
# Without previous interaction
# =====================================================

model_keyword_1 = fit_sklearn_linear_model(
    df=df,
    feature_cols=features_without_previous_interaction,
    target_col="sycophancy_keyword_count",
    model_name="C: keyword count, no previous interaction"
)

print("\n\n=====================================================")
print("REGRESSION C")
print("Target: sycophancy_keyword_count")
print("Predictors: LLM Neutral, Support, Entailment + agent model")
print("=====================================================")

print("\nModel metrics:")
print(model_keyword_1["metrics"].to_string(index=False))

print("\nRegression results:")
print(model_keyword_1["results"].to_string(index=False))


# =====================================================
# Regression D
# Target: sycophancy_keyword_count
# With previous interaction
# =====================================================

model_keyword_2 = fit_sklearn_linear_model(
    df=df,
    feature_cols=features_with_previous_interaction,
    target_col="sycophancy_keyword_count",
    model_name="D: keyword count, with previous interaction"
)

print("\n\n=====================================================")
print("REGRESSION D")
print("Target: sycophancy_keyword_count")
print("Predictors: LLM Neutral, Support, Entailment + agent model + previous interaction")
print("=====================================================")

print("\nModel metrics:")
print(model_keyword_2["metrics"].to_string(index=False))

print("\nRegression results:")
print(model_keyword_2["results"].to_string(index=False))


# =====================================================
# Model comparison
# =====================================================

comparison = pd.concat(
    [
        model_likelihood_1["metrics"],
        model_likelihood_2["metrics"],
        model_keyword_1["metrics"],
        model_keyword_2["metrics"]
    ],
    ignore_index=True
)

print("\n\n=====================================================")
print("MODEL COMPARISON")
print("=====================================================")
print(comparison.to_string(index=False))


# =====================================================
# Save results to Excel
# =====================================================

output_folder = r"C:\Users\capod\Downloads\Dataset_NLP"
excel_path = output_folder + r"\sycophancy_sklearn_regression_results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    model_likelihood_1["results"].to_excel(
        writer,
        sheet_name="A_likelihood_no_prev",
        index=False
    )

    model_likelihood_2["results"].to_excel(
        writer,
        sheet_name="B_likelihood_prev",
        index=False
    )

    model_keyword_1["results"].to_excel(
        writer,
        sheet_name="C_keyword_no_prev",
        index=False
    )

    model_keyword_2["results"].to_excel(
        writer,
        sheet_name="D_keyword_prev",
        index=False
    )

    comparison.to_excel(
        writer,
        sheet_name="Model_comparison",
        index=False
    )

    model_likelihood_1["predictions"].to_excel(
        writer,
        sheet_name="Pred_likelihood_no_prev",
        index=False
    )

    model_likelihood_2["predictions"].to_excel(
        writer,
        sheet_name="Pred_likelihood_prev",
        index=False
    )

    model_keyword_1["predictions"].to_excel(
        writer,
        sheet_name="Pred_keyword_no_prev",
        index=False
    )

    model_keyword_2["predictions"].to_excel(
        writer,
        sheet_name="Pred_keyword_prev",
        index=False
    )

print("\nExcel file saved successfully:")
print(excel_path)