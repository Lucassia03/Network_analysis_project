# Sycophancy in Online AI-Assisted Discourse

A descriptive and non-parametric analysis of linguistic validation patterns in AI-agent discourse.

## Overview

This project studies **sycophantic language** in a corpus of online posts and comments from an AI-focused community. The analysis asks whether sycophancy is best explained by semantic agreement, thematic context, predicted model identity, or local conversational structure.

The main finding is that sycophancy is **not simply semantic agreement**. Comments classified as semantically neutral show the highest levels of sycophantic language, while entailment comments show the lowest. The evidence instead suggests that sycophancy functions as a pragmatic style of social validation shaped by topic, discourse context, and conversational anchoring.

## Research Question

**What influences sycophancy in AI-agent discourse?**

The project evaluates three main explanations:

1. **Semantic relation**: whether comments entail, contradict, or are neutral toward parent posts.
2. **Thematic context**: whether specific topics or clusters contain more validating language.
3. **Predicted model identity**: whether sycophancy varies across comments attributed to ChatGPT, Claude, DeepSeek, and Llama.

A complementary LLM-as-a-judge analysis examines whether local context, parent alignment, and conversational anchoring explain sycophancy better than broad agreement with the original post.

## Dataset

The dataset contains:

| Component | Count |
|---|---:|
| Posts | 822 |
| Comments | 24,563 |
| Full merged non-parametric dataset | 23,492 comments |
| Stance subset | 9,253 comments |
| Rich LLM-as-a-judge subset | 1,500 comments |

Each comment is linked to a parent post or parent comment, enabling both semantic and conversational analyses.

## Main Variables

### Outcome variables

| Variable | Description |
|---|---|
| `sycophancy_keyword_count` | Count of validation, praise, agreement, or positive-reinforcement keywords in a comment. |
| `sycophancy_intensity` | Categorical version of keyword count: 0, 1, or 2+ keywords. |
| `final_sycophancy_likelihood` | Continuous LLM-as-a-judge score measuring contextual likelihood of sycophancy. |
| `final_sycophancy_label` | Categorical LLM-as-a-judge label: no clear, weak, or moderate sycophancy signal. |

### Predictor variables

| Variable family | Examples |
|---|---|
| NLI labels | `Neutral`, `Entailment`, `Contradiction` |
| NLI scores | neutral, entailment, contradiction scores |
| Similarity | cosine similarity between comment and parent post |
| Topic structure | TF-IDF clusters, NMF topics |
| Model attribution | predicted ChatGPT, Claude, DeepSeek, Llama labels |
| LLM-as-a-judge context | post stance, parent stance, context alignment, tone, depth |
| Interaction history | `had_previous_interaction` |

## Methods

The analysis combines:

- Natural Language Inference (NLI)
- Cosine similarity
- TF-IDF clustering
- Non-negative Matrix Factorisation (NMF)
- Predicted model attribution
- LLM-as-a-judge stance and context analysis
- Manual validation of stance/NLI labels
- Non-parametric inference
- Reduced regression checks

### Statistical tests

The main inferential tests are non-parametric because `sycophancy_keyword_count` is skewed and zero-heavy:

- Kruskal-Wallis tests
- Mann-Whitney U tests with Benjamini-Hochberg correction
- Chi-square tests on sycophancy intensity categories
- Spearman rank correlations
- Permutation tests
- Reduced linear/logistic regression checks

## Key Findings

### 1. Sycophancy is not semantic agreement

If sycophancy were simply agreement, entailment comments should have the highest sycophancy. Instead, the ordering is:

```text
Neutral > Contradiction > Entailment
```

| NLI label | Mean keyword count | % any sycophancy | % 2+ keywords |
|---|---:|---:|---:|
| Neutral | 0.654 | 42.36% | 15.50% |
| Contradiction | 0.547 | 36.40% | 13.09% |
| Entailment | 0.417 | 26.00% | 9.58% |

This suggests that sycophancy is better understood as a pragmatic and social function rather than as logical agreement.

### 2. Topic is the strongest keyword-based predictor

NMF topic membership explains about **10.4%** of rank-based variation in sycophancy keyword count.

| Topic | Topic terms | % any sycophancy | % 2+ keywords |
|---:|---|---:|---:|
| 10 | great, great question, skills, goal | 87.55% | 58.50% |
| 2 | locivault, browser, fly dev | 72.93% | 43.39% |
| 8 | sharing, thanks, perspective | 66.55% | 31.09% |
| 5 | consistency, gossip, writes, eventual | 3.54% | 0.53% |

Sycophancy is concentrated in promotional, evaluative, career-oriented, and socially validating topics, and is much lower in technical or security-oriented topics.

### 3. Predicted model identity is significant but secondary

In the full merged dataset, ChatGPT-attributed comments are substantially less sycophantic than Claude-, DeepSeek-, and Llama-attributed comments.

| Predicted model | N | Mean keyword count | % any sycophancy | % 2+ keywords |
|---|---:|---:|---:|---:|
| ChatGPT | 13,583 | 0.437 | 31.00% | 9.10% |
| Claude | 4,043 | 0.800 | 51.00% | 19.12% |
| DeepSeek | 4,789 | 0.858 | 50.22% | 22.64% |
| Llama | 1,077 | 0.917 | 52.46% | 26.09% |

The model-level effect is smaller than the topic effect but larger than the NLI effect.

### 4. Local conversational context matters more than post-level support

The LLM-as-a-judge analysis shows that sycophancy is not mainly driven by support for the original post. Instead, it is associated with:

- parent alignment
- context anchoring
- tone
- conversation depth
- previous interaction

Post stance is not significantly associated with the final sycophancy label, while parent stance and context alignment are strongly associated.

### 5. Previous interaction improves reduced regression models

Adding `had_previous_interaction` improves model fit for both outcomes.

| Specification | Target | R² without previous interaction | R² with previous interaction | Previous-interaction coefficient |
|---|---|---:|---:|---:|
| Categorical stance | `final_sycophancy_likelihood` | 0.0217 | 0.1164 | +0.0351 |
| Categorical stance | `sycophancy_keyword_count` | 0.1754 | 0.2101 | +0.2161 |
| Continuous NLI scores | `final_sycophancy_likelihood` | 0.0089 | 0.1122 | +0.0367 |
| Continuous NLI scores | `sycophancy_keyword_count` | 0.1690 | 0.2058 | +0.2230 |

This supports the interpretation that sycophancy is shaped by local conversational history and context anchoring.

## Interpretation

The results converge on a common mechanism: sycophancy in this corpus is best understood as **local conversational accommodation**.

It is shaped by:

- topic and thematic norms
- agent or author style
- parent-child alignment
- previous interaction
- context anchoring
- pragmatic validation language

It is **not reducible to semantic entailment** or broad agreement with the original post.

## Limitations

- Keyword counts are proxies and may miss non-lexical forms of sycophancy.
- Predicted model labels are classifier outputs, not ground-truth model identities.
- The analysis is observational and should not be interpreted causally.
- LLM-as-a-judge labels are prompt-sensitive and available only on subsets of the data.
- Some strong LLM-as-a-judge predictors are components of the same scoring pipeline that produces the final likelihood score.

## Suggested Repository Structure

```text
.
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── notebooks/
│   ├── 01_nli_and_similarity.ipynb
│   ├── 02_topic_modeling.ipynb
│   ├── 03_model_attribution.ipynb
│   ├── 04_llm_as_judge_analysis.ipynb
│   └── 05_regression_checks.ipynb
├── scripts/
│   ├── build_features.py
│   ├── run_nonparametric_tests.py
│   ├── run_regressions.py
│   └── export_tables.py
├── results/
│   ├── tables/
│   ├── figures/
│   └── regression_outputs/
└── paper/
    └── manuscript.docx
```

## Reproducibility Notes

The analyses require comment-level data containing:

- comment text
- parent post/comment identifiers
- NLI labels and scores
- cosine similarity scores
- TF-IDF cluster labels
- NMF topic labels
- predicted model labels
- sycophancy keyword counts
- LLM-as-a-judge annotations where available

Because the corpus may contain scraped user-generated or AI-generated discourse, repository maintainers should verify whether the raw data can be publicly released. If not, provide anonymised or aggregated outputs only.

## Citation

If you use or adapt this analysis, cite the related work below.

```text
Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019, 4171-4186.

Perez, E., Ringer, S., Lukosiute, K., et al. 2023. Discovering Language Model Behaviors with Model-Written Evaluations. Findings of the Association for Computational Linguistics: ACL 2023.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., et al. 2023. Towards Understanding Sycophancy in Language Models. arXiv:2310.13548.

Vennemeyer, D., Duong, P. A., Zhan, T., and Jiang, T. 2025. Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs. arXiv:2509.21305.

Noshin, K., Ahmed, S. I., and Sultana, S. 2026. User Detection and Response Patterns of Sycophantic Behavior in Conversational AI. arXiv:2601.10467.
```

## License

Add a license before publishing this repository. For academic code and tables, common options include MIT, Apache-2.0, or CC BY 4.0 for documentation.
