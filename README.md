# Sycophancy in Online AI-Assisted Discourse:

*A Descriptive and Non-Parametric Analysis of Linguistic Validation Patterns*

## Abstract

This study examines sycophantic language patterns in a large corpus of online posts and comments originating from a community engaged with AI-related topics. Drawing on a dataset of 822 posts and 24,563 comments, we operationalise sycophancy through a keyword-based scoring framework and relate it to three sets of covariates: (i) Natural Language Inference (NLI) labels and cosine similarity scores measuring the semantic relationship between comments and parent posts; (ii) TF-IDF-derived thematic clusters and NMF topic assignments capturing the subject matter of comments; and (iii) predicted AI model authorship labels across the full set of 23,492 observations. Non-parametric tests — including Kruskal-Wallis, Mann-Whitney U, chi-square, Spearman correlations, and permutation analyses — consistently indicate that sycophancy is strongly associated with thematic content and significantly associated with predicted model identity, and weakly but significantly associated with NLI category. Crucially, sycophancy is highest in semantically neutral comments rather than in those that logically entail the parent post, suggesting that sycophantic language functions primarily as a pragmatic style of social validation rather than as semantic agreement. At the model level, ChatGPT-attributed comments are substantially less sycophantic than those attributed to Claude, DeepSeek, and Llama, a difference that is robust across all tests. These findings have implications for the study of AI-generated text, online communication dynamics, and the measurement of sycophancy as a communicative phenomenon.

## 1. Introduction and Descriptive Overview

The concept of sycophancy — broadly defined as the tendency to produce excessively agreeable, validating, or flattering responses — has emerged as a central concern in research on large language models (LLMs). While it has been extensively discussed in the context of AI alignment and RLHF-induced biases, far less attention has been devoted to studying sycophancy as a naturally occurring communicative pattern in human-AI or human-in-loop online discourse. This paper addresses that gap by examining sycophantic language in a corpus of posts and comments scraped from an online community centred on AI-related topics.

Our point of departure is empirical: we wish to understand how sycophantic language is distributed across the conversational space of this community, what structural and semantic features of the discourse are associated with it, and whether it can be reduced to simpler constructs such as semantic similarity or entailment. In doing so, we bring together three distinct analytical frameworks — Natural Language Inference (NLI), topic modelling via TF-IDF clustering and Non-negative Matrix Factorisation (NMF), and predicted model classification — and subject their relationship with sycophancy to rigorous non-parametric testing.

This section introduces the dataset, defines the core measurement concepts employed throughout the paper, and provides a first descriptive view of the data. Subsequent sections describe the methodology in detail and present the results of the non-parametric analyses. The paper concludes with a synthetic discussion of what the evidence collectively implies about the determinants of sycophancy in this corpus.

### 1.1 The Dataset

The dataset was constructed by scraping posts and comments from an online community focused on discussions about AI agents, memory systems, distributed computing, and related topics. The full dataset comprises 822 posts and 24,563 comments replying to those posts. Each comment is linked to its parent post through a parent-child relationship, which forms the basis for the semantic and textual analyses described below.

The dataset is rich in structure: comments carry information about their authorship, their thematic content (as assigned by topic modelling), their semantic relationship with the parent post (as measured by NLI and cosine similarity), and a predicted AI model label indicating whether the comment is attributed to ChatGPT, Claude, Llama, or DeepSeek. Model labels are available for 23,492 comments after merging with the sycophancy dataset, providing near-complete coverage of the corpus and substantially stronger statistical power for the model-level analysis than would be available from a smaller labelled subset.

### 1.2 Natural Language Inference: Concept and Distribution

Natural Language Inference (NLI) is a computational task that classifies the directional semantic relationship between two pieces of text — typically called the premise and the hypothesis. Given a parent post (premise) and a comment (hypothesis), an NLI model assigns one of three labels:

- **Entailment:** the comment follows logically or semantically from the post. Its content is supported by, consistent with, or a direct consequence of what the parent post asserts.

- **Contradiction:** the comment is in semantic conflict with the post. It asserts something that cannot simultaneously be true with the post's content.

- **Neutral:** the comment neither follows from nor contradicts the post. The two texts are semantically unrelated or only loosely connected.

Importantly, NLI is a measure of logical or semantic relatedness, not of communicative tone or social function. A comment can be polite, validating, and agreeable in tone while still being logically neutral with respect to the parent post — and this distinction will prove critical in our analysis of sycophancy.

In our corpus, the NLI label distribution across the 24,563 parent-child comment pairs is as follows:

| **NLI Prediction** | **Count** | **Percentage (%)** |
|--------------------|-----------|--------------------|
| Neutral            | 18,534    | 75.45              |
| Contradiction      | 3,541     | 14.42              |
| Entailment         | 2,488     | 10.13              |

*Table 1. Distribution of NLI labels across 24,563 comment-post pairs.*

The vast majority of comments — 75.45% — are classified as neutral with respect to their parent post. This reflects the nature of online discourse, where most comments neither strictly follow from nor contradict the parent post, but instead introduce new topics, share opinions, or engage socially without altering the logical content of the conversation. Contradiction is more common than entailment, accounting for 14.42% of pairs compared to 10.13% for entailment, suggesting that disagreement and contrast are more frequent than strict logical support in this community.

### 1.3 Cosine Similarity: Concept and Distribution by NLI Label

Cosine similarity measures the degree of textual or semantic overlap between two documents, regardless of their logical relationship. It is computed as the cosine of the angle between two vector representations of the texts — typically derived from word embeddings or TF-IDF transformations. A value of 1 indicates that the two texts are identical in their vector representation; a value of 0 indicates complete orthogonality (no shared content); negative values, which can arise depending on the embedding method, indicate that the texts are in some sense opposed in the vector space.

Unlike NLI, cosine similarity is a symmetric, magnitude-free measure of shared content. Two texts can share many of the same words and topics — and thus have high cosine similarity — while still contradicting each other semantically. Conversely, two texts can have low cosine similarity because they use different vocabulary, even if one logically entails the other.

The distribution of cosine similarity scores across NLI labels in our corpus is reported below:

| **NLI Prediction** | **Mean Similarity** | **Std Dev** | **Median** | **Min** | **Max** | **Count** |
|--------------------|---------------------|-------------|------------|---------|---------|-----------|
| Contradiction      | 0.3525              | 0.2227      | 0.3740     | −0.1441 | 0.9144  | 3,541     |
| Entailment         | 0.4406              | 0.2256      | 0.4926     | −0.1212 | 1.0000  | 2,488     |
| Neutral            | 0.4611              | 0.1934      | 0.4945     | −0.1246 | 0.9776  | 18,534    |

*Table 2. Descriptive statistics of cosine similarity by NLI label.*

Several features of this distribution are noteworthy. First, neutral comments have the highest mean cosine similarity (0.4611), which might seem paradoxical: one might expect entailment comments to share more surface content with the parent post. However, this pattern is consistent with the character of neutral discourse: neutral comments often mirror the vocabulary and frame of the parent post while adding unrelated or only loosely connected content, resulting in high lexical overlap without logical dependency. Second, contradiction comments have the lowest mean cosine similarity (0.3525), consistent with the idea that contradictory comments often use different vocabulary to frame an opposing argument. Third, the overlap between the distributions is large in all cases, as shown by the wide standard deviations and the broad range of values, indicating that cosine similarity and NLI label are far from redundant measures.

### 1.4 Sycophancy: Definition, Operationalisation, and Measurement

For the purposes of this study, sycophancy is defined as the presence of lexically expressed validation, praise, agreement, or positive reinforcement in a comment. This definition is deliberately pragmatic rather than psychological: we do not attempt to infer intent or attitude, but instead operationalise sycophancy through the occurrence of a predefined set of sycophancy-related keywords — expressions such as "great," "exactly," "right," "interesting," "great question," and similar terms that signal approving, validating, or agreement-oriented language.

Two core measures are constructed from this keyword list:

1.  **Sycophancy keyword count (***sycophancy_keyword_count***):** the total number of sycophancy-related keywords appearing in a comment. This is the primary outcome variable in all statistical analyses.

2.  **Sycophancy intensity categories:** a categorical version of the keyword count, distinguishing between comments with no sycophancy keywords (count = 0), low sycophancy (count = 1), and higher sycophancy (count ≥ 2). Used in chi-square tests and for descriptive comparisons.

A cluster-level composite sycophancy score is also constructed for the TF-IDF clustering analysis. This score combines three components: the comment hit rate (the proportion of comments in a cluster containing at least one sycophancy keyword), the keyword hits per 100 comments (measuring the density of sycophantic keywords within the cluster), and a top-term overlap score (measuring how much the cluster's most frequent keywords overlap with the canonical list of sycophantic terms). Formally:

*Sycophancy Score = Comment Hit Rate (%) + Keyword Hits per 100 Comments + Top-Term Overlap Score*

This composite score captures both the extensive margin of sycophancy — how many comments contain any validating language — and the intensive margin — how frequently and densely such language appears. It is important to emphasise that this score is a keyword-based proxy rather than a direct psychological measurement. Expressions such as "right" or "interesting" can appear in neutral or substantive contexts, and the score should be understood as a measure of the prevalence of sycophancy-like language, not a guarantee of intentional flattery in each individual comment.

In the full merged dataset of 23,492 comment-level observations used for the non-parametric analyses, the distribution of sycophancy_keyword_count is strongly right-skewed and concentrated at low values: 14,262 comments contain no sycophancy keyword, 5,856 contain exactly one, 2,214 contain two, and only a small minority contain three or more. This distributional shape — bounded, discrete, and highly skewed — motivates the non-parametric methodological approach adopted throughout the paper.

## 2. Thematic Structure, Author Specialisation, and Sycophancy by Cluster

Before turning to the non-parametric analyses of sycophancy, it is important to characterise the thematic structure of the corpus. The comments in the dataset are not drawn from an undifferentiated conversational space; rather, they are organised around distinct topics and communities of interest. Understanding this structure is essential for interpreting the variation in sycophancy observed across the corpus.

### 2.1 TF-IDF Cluster Structure

The dataset was thematically partitioned using Term Frequency-Inverse Document Frequency (TF-IDF) clustering, which groups documents on the basis of their most distinctive vocabulary. This procedure produced seven clusters, each characterised by a descriptive label derived from the most frequent and discriminating terms:

| **Cluster ID** | **Label**                                  |
|----------------|--------------------------------------------|
| 0              | onchain_coinflip_promo                     |
| 1              | distributed_memory_storage_latency         |
| 2              | agent_governance_trust_security            |
| 3              | general_memory_identity_systems_discussion |
| 4              | locivault_private_encrypted_memory         |
| 5              | distributed_consistency_gossip_sync        |
| 6              | resume_career_narrative_skills             |

*Table 3. TF-IDF cluster identifiers and labels.*

The clusters span a wide range of topics: from highly technical discussions about distributed systems, memory storage, and consistency protocols (clusters 1, 5) to more social or personal topics such as career development and identity (cluster 6) and speculative or promotional content about blockchain-related coinflip mechanisms (cluster 0). This diversity is important because, as we will see, sycophancy is not uniformly distributed across this thematic landscape.

### 2.2 Author Specialisation Across Clusters

An important structural feature of the corpus is that authors tend to be highly specialised in their cluster participation: most authors concentrate the vast majority of their comments in a single cluster rather than engaging broadly across multiple topic areas. This finding is relevant for interpreting sycophancy variation, since it implies that the observed differences in sycophancy across clusters are not simply artefacts of individual-level variation in authoring style — they reflect genuine differences in the communicative norms and linguistic conventions of distinct thematic communities.

The dataset contains 1,796 unique authors. Of these, 1,646 have at least one clustered comment, while the remaining 150 are excluded from the specialisation analysis. Among active authors, 65.2% appear in only one cluster, 33.0% in two clusters, 1.6% in three clusters, and just 0.1% in four clusters.

| **Clusters Participated In** | **Number of Authors** | **Share of Active Authors (%)** |
|------------------------------|-----------------------|---------------------------------|
| 1                            | 1,074                 | 65.2                            |
| 2                            | 543                   | 33.0                            |
| 3                            | 27                    | 1.6                             |
| 4                            | 2                     | 0.1                             |

*Table 4. Author-level cluster participation.*

This concentration is not simply a mechanical consequence of low activity. Even after restricting the sample to authors with at least 10, 20, or 50 clustered comments, the mean dominant-cluster share remains near or above 0.848, and at least half of active authors have 90% or more of their comments in a single cluster. The Herfindahl-Hirschman Index (HHI) and effective number of clusters corroborate this picture: the typical active author is effectively present in just one thematic cluster.

| **Min. Clustered Comments** | **Authors** | **Mean Dominant-Cluster Share** | **Share ≥ 90%** | **Share ≥ 80%** | **Share ≥ 70%** |
|-----------------------------|-------------|---------------------------------|-----------------|-----------------|-----------------|
| ≥ 1                         | 1,646       | 0.906                           | 71.0%           | 78.3%           | 84.6%           |
| ≥ 5                         | 645         | 0.847                           | 49.1%           | 67.6%           | 77.7%           |
| ≥ 10                        | 379         | 0.848                           | 53.0%           | 67.8%           | 78.9%           |
| ≥ 20                        | 219         | 0.849                           | 52.5%           | 68.5%           | 80.4%           |
| ≥ 50                        | 92          | 0.865                           | 54.3%           | 73.9%           | 83.7%           |

*Table 5. Author specialisation across activity thresholds.*

### 2.3 Sycophancy Scores by Cluster

The cluster-level sycophancy scores reveal strong heterogeneity in the prevalence of validating language across thematic areas. The results are reported in the table below.

| **Rank** | **Cluster** | **Label**                           | **Comments** | **Hit Rate (%)** | **KW Hits/100** | **Overlap** | **Score** | **Top Keywords**            |
|----------|-------------|-------------------------------------|--------------|------------------|-----------------|-------------|-----------|-----------------------------|
| 1        | 0           | onchain_coinflip_promo              | 305          | 95.08            | 204.59          | 0.00        | 299.67    | right, exactly, interesting |
| 2        | 6           | resume_career_narrative_skills      | 7,495        | 46.10            | 71.90           | 8.00        | 126.00    | right, exactly, interesting |
| 3        | 1           | distributed_memory_storage_latency  | 2,083        | 45.18            | 71.10           | 0.00        | 116.27    | right, exactly, true        |
| 4        | 3           | general_memory_identity_systems     | 1,335        | 38.58            | 70.49           | 0.00        | 109.06    | great, great question, good |
| 5        | 5           | distributed_consistency_gossip_sync | 10,020       | 36.48            | 54.22           | 7.00        | 97.70     | right, interesting, true    |
| 6        | 4           | locivault_private_encrypted_memory  | 700          | 28.14            | 31.57           | 1.00        | 60.71     | interesting, exactly, right |
| 7        | 2           | agent_governance_trust_security     | 1,554        | 11.39            | 11.52           | 0.00        | 22.91     | true, exactly, correct      |

*Table 6. Sycophancy scores by TF-IDF cluster (ranked).*

Cluster 0 (onchain_coinflip_promo) is a striking outlier. With a sycophancy score of 299.67, it is more than twice as sycophantic as the second-ranked cluster. Its comment hit rate of 95.08% means that virtually every comment contains at least one sycophancy keyword, and with 204.59 keyword hits per 100 comments, the density of sycophantic language exceeds two occurrences per comment on average. The promotional, speculative, and highly social nature of this cluster — centred on blockchain-related coinflip mechanics — appears to generate an environment of pervasive validation and positive reinforcement.

Clusters 6, 1, and 3 form a second tier of moderately high sycophancy, each with comment hit rates around 38–46% and keyword densities near 70 per 100 comments. The character of sycophancy differs subtly across these clusters: in cluster 6 (career and identity discussions), high overlap with canonical sycophancy terms and the predominance of expressions like "right," "exactly," and "interesting" suggest a conversational style built on validation and encouragement. In cluster 3, the top keywords shift toward more overtly evaluative terms — "great," "great question," and "good" — indicating a style of explicit praise rather than mere agreement.

At the opposite extreme, cluster 2 (agent_governance_trust_security) is the least sycophantic, with only 11.39% of comments containing any sycophancy keyword and a score of 22.91. The evaluative, analytical, and security-focused character of this cluster appears to foster a more critical and sceptical discursive norm, where validation language is comparatively rare. This finding suggests that sycophancy is not merely a stylistic idiosyncrasy of individual authors, but is systematically shaped by the topical and normative context of the discussion community.

A methodologically important distinction emerges from this analysis: relative sycophancy (the proportion of comments in a cluster containing sycophantic language) and absolute sycophancy volume (the total number of keyword hits contributed by a cluster) do not co-vary perfectly. Cluster 5 (distributed_consistency_gossip_sync) ranks fifth on the relative score but contributes the largest absolute number of keyword hits in the dataset, owing to its size of over 10,000 comments. This distinction matters when interpreting what drives aggregate sycophancy in the corpus as a whole versus what makes individual thematic communities more or less characterised by validating discourse.

## 3. Non-Parametric Analysis: Methodology and Results

The core statistical analyses of this paper adopt a non-parametric approach. This choice is motivated by the distributional characteristics of the primary outcome variable, sycophancy_keyword_count: it is a bounded discrete count variable, heavily right-skewed, and concentrated at zero. Standard parametric tests — which assume approximate normality and homogeneity of variance — are therefore inappropriate. Instead, we employ a suite of four complementary methods: the Kruskal-Wallis test, pairwise Mann-Whitney U tests with Benjamini-Hochberg false discovery rate correction, chi-square tests on intensity categories, and permutation tests. Together, these methods provide a robust and multi-faceted assessment of the association between sycophancy and each set of covariates.

### 3.1 Non-Parametric Tests: Overview

The Kruskal-Wallis test is a rank-based generalisation of the one-way ANOVA. It tests whether the distributions of a continuous or ordinal outcome variable are identical across K independent groups, without assuming normality. The test statistic H approximately follows a chi-square distribution under the null hypothesis of distributional equality. We supplement the test with epsilon-squared (ε²) as a non-parametric effect size, computed as (H − k + 1) / (n − k), where k is the number of groups and n is the number of observations.

When the Kruskal-Wallis test rejects the null hypothesis, we follow up with pairwise Mann-Whitney U tests to identify which specific group pairs differ. Multiple testing is addressed using the Benjamini-Hochberg false discovery rate (FDR) procedure, which controls the expected proportion of false discoveries among all rejected hypotheses — a less conservative approach than Bonferroni correction, well-suited to exploratory settings with many comparisons.

The chi-square test on sycophancy intensity categories — constructed by collapsing sycophancy_keyword_count into three levels (0, 1, ≥ 2) — provides a complementary perspective by examining the association between group membership and sycophancy intensity without any distributional assumption. Cramér's V is computed as a standardised effect size. The permutation test provides a model-free robustness check: by randomly permuting group labels 1,000 times and comparing the observed between-group variance in rank-transformed sycophancy counts to the simulated null distribution, it tests whether the observed group differences are unlikely under random assignment.

### 3.2 Sycophancy Across NMF Topics

Non-negative Matrix Factorisation (NMF) is a dimensionality reduction technique that decomposes a document-term matrix into a smaller number of latent topic dimensions. Unlike TF-IDF clustering — which assigns documents to clusters on the basis of distinctive vocabulary — NMF produces a probabilistic topic-document matrix in which each document receives a weight for each topic. In our analysis, each comment is assigned to its dominant NMF topic. The NMF analysis was conducted on a merged dataset of 23,481 comments across 15 topic classes.

#### 3.2.1 Descriptive Evidence

| **Topic** | **N** | **Topic Terms**                           | **% Any Sycophancy** | **% 2+ Keywords** |
|-----------|-------|-------------------------------------------|----------------------|-------------------|
| 10        | 506   | great, great question, skills, goal       | 87.55                | 58.50             |
| 2         | 484   | locivault, browser, fly dev               | 72.93                | 43.39             |
| 8         | 1,187 | sharing, thanks, perspective              | 66.55                | 31.09             |
| 0         | 5,517 | something, system, know, question         | 43.61                | 16.59             |
| 14        | 2,676 | memory, file, drift, continuity           | 42.83                | 16.29             |
| 4         | 310   | chain, usdc, battle, coinflip             | 39.03                | 4.84              |
| 13        | 2,361 | cost, agents, heartbeat, carbon           | 37.36                | 10.93             |
| 9         | 5,751 | agent, human, governance, trust, security | 35.66                | 11.51             |
| 7         | 774   | fork, identity, test, path, copy          | 35.27                | 11.24             |
| 11        | 473   | latency, storage, distributed, usb, bus   | 35.10                | 3.17              |
| 3         | 1,472 | role, resume, older, line, value          | 25.54                | 5.10              |
| 6         | 577   | agentflex, vip, leaderboard               | 24.26                | 5.20              |
| 1         | 403   | lan sub, authenticated reads              | 9.68                 | 0.50              |
| 12        | 425   | multicast, lan authenticated, memory      | 4.94                 | 0.00              |
| 5         | 565   | consistency, gossip, writes, eventual     | 3.54                 | 0.53              |

*Table 7. Sycophancy rates by NMF topic (ranked by % with any sycophancy).*

Topic 10, characterised by terms such as "great," "great question," and "goal," has the highest sycophancy rate by a wide margin: 87.55% of its comments contain at least one sycophancy keyword, and 58.50% contain two or more. This topic appears to represent a space of explicit encouragement, mentorship, and evaluative praise. Topics 2 and 8 also show high sycophancy rates (72.93% and 66.55% respectively), consistent with the community-oriented and socially connective character of their content. At the opposite end, topics 5, 12, and 1 — centred on technical content such as consistency protocols, multicast authentication, and LAN reads — have sycophancy rates below 10%, consistent with a more analytical and substantive discursive register.

#### 3.2.2 Statistical Tests

The Kruskal-Wallis test strongly rejects the null hypothesis of distributional equality across NMF topics:

*H = 2441.874, p \< 0.001, ε² = 0.1035*

The effect size ε² = 0.1035 indicates that topic membership accounts for approximately 10.4% of the rank-based variation in sycophancy keyword counts. By non-parametric standards, this is a meaningful and substantively relevant association, not merely a statistically detectable one.

Pairwise Mann-Whitney U tests with Benjamini-Hochberg FDR correction confirm broad separation across topics: topic 10 differs significantly from topics 3, 9, 5, 13, 12, 1, 0, 6, 14, 7, 11, and 4. Topic 8 similarly differs significantly from low-sycophancy topics including 5, 3, 9, 12, 1, 13, and 6. These findings establish that the global Kruskal-Wallis result is not driven by a single outlier comparison, but reflects a broad structural gradient from highly sycophantic to minimally sycophantic topics.

The chi-square test on sycophancy intensity categories further rejects independence:

*χ²(28) = 2787.227, p \< 0.001, Cramér's V = 0.244*

A Cramér's V of 0.244 indicates a moderate association between topic membership and sycophancy intensity. Finally, the permutation test confirms that the observed between-topic rank variance of 3,631,193.47 is extremely unlikely under random topic assignment, with a permutation p-value of 0.001.

### 3.3 Sycophancy Across Predicted AI Models

This analysis examines whether sycophancy varies systematically across predicted AI model labels. Unlike earlier versions of this analysis that were restricted to a small labelled subset, the full merged dataset of 23,492 comments is used here, providing near-complete coverage of the corpus. The model distribution is: ChatGPT (n = 13,583), DeepSeek (n = 4,789), Claude (n = 4,043), and Llama (n = 1,077). Before the merge, the model-level database recorded 24,563 observations; after merging with the sycophancy dataset, 23,492 matched observations are retained, with very similar proportions across models.

This analysis is of direct relevance for understanding whether different AI systems — or rather, the textual signatures attributed to them by the classification model — exhibit systematically different propensities for validating, praising, or agreement-oriented language. With a substantially larger and more representative sample than previously available, the statistical power to detect even modest model-level differences is high.

#### 3.3.1 Descriptive Evidence

| **Predicted Model** | **N**  | **Mean KW Count** | **Median** | **75th Pct.** | **90th Pct.** | **% Any Sycophancy** | **% 2+ Keywords** | **Avg. Confidence** |
|---------------------|--------|-------------------|------------|---------------|---------------|----------------------|-------------------|---------------------|
| ChatGPT             | 13,583 | 0.437             | 0.0        | 1.0           | 1.0           | 31.00                | 9.10              | 0.960               |
| Claude              | 4,043  | 0.800             | 1.0        | 1.0           | 2.0           | 51.00                | 19.12             | 0.896               |
| DeepSeek            | 4,789  | 0.858             | 1.0        | 1.0           | 2.0           | 50.22                | 22.64             | 0.926               |
| Llama               | 1,077  | 0.917             | 1.0        | 2.0           | 3.0           | 52.46                | 26.09             | 0.922               |

*Table 8. Sycophancy descriptive statistics by predicted AI model (full merged dataset, n = 23,492).*

The descriptive picture is striking and unambiguous. ChatGPT-attributed comments have a mean sycophancy keyword count of 0.437 — less than half the mean of Claude (0.800), DeepSeek (0.858), or Llama (0.917). The difference is even sharper when looking at the upper end of the distribution: while Llama's 90th percentile is 3 keywords and Claude's and DeepSeek's are 2, ChatGPT's 90th percentile is only 1. The share of comments with two or more sycophancy keywords — the higher-intensity tier — is 9.10% for ChatGPT, compared to 19.12% for Claude, 22.64% for DeepSeek, and 26.09% for Llama. In other words, Llama-attributed comments are nearly three times as likely as ChatGPT-attributed comments to contain multiple sycophancy keywords.

The median keyword count is 0 for ChatGPT, meaning the majority of ChatGPT-attributed comments contain no sycophancy keywords at all — a profile very different from Claude, DeepSeek, and Llama, where the median is 1. Average prediction confidence is high for all models, ranging from 0.896 (Claude) to 0.960 (ChatGPT), suggesting that most attributions are made with substantial certainty by the classification model. The ChatGPT group is also by far the largest, with 13,583 observations, providing ample statistical power for any comparison involving it.

#### 3.3.2 Statistical Tests

To formally test whether the distribution of sycophancy_keyword_count differs across predicted models, I use a Kruskal-Wallis test. The null hypothesis is that the distribution of sycophancy keyword counts is identical across ChatGPT, Claude, DeepSeek, and Llama.

The test strongly rejects the null hypothesis:

*H = 1113.954, p \< 0.001, ε² = 0.0473*

The effect size ε² = 0.0473 indicates that predicted model identity accounts for approximately 4.7% of the rank-based variation in sycophancy keyword counts. This is a substantively meaningful association: it is smaller than the topic-level effect (ε² = 0.1035), but considerably larger than the NLI-level effect (ε² = 0.0129). In other words, knowing which model produced a comment explains more about its sycophancy level than knowing its NLI category, although topic remains the strongest single predictor.

Pairwise Mann-Whitney U tests with Benjamini-Hochberg FDR correction reveal the structure of the model-level differences:

| **Comparison**      | **Median (Group 1)** | **Median (Group 2)** | **FDR-adj. p-value** | **Significant at 5%?** |
|---------------------|----------------------|----------------------|----------------------|------------------------|
| ChatGPT vs Claude   | 0.0                  | 1.0                  | \< 0.001             | Yes                    |
| ChatGPT vs DeepSeek | 0.0                  | 1.0                  | \< 0.001             | Yes                    |
| ChatGPT vs Llama    | 0.0                  | 1.0                  | \< 0.001             | Yes                    |
| Claude vs Llama     | 1.0                  | 1.0                  | 0.007                | Yes                    |
| DeepSeek vs Llama   | 1.0                  | 1.0                  | 0.052                | No                     |
| Claude vs DeepSeek  | 1.0                  | 1.0                  | 0.223                | No                     |

*Table 9. Pairwise Mann-Whitney tests: sycophancy by predicted model (FDR-corrected).*

The results reveal a clear two-tier structure. ChatGPT is significantly less sycophantic than all three other models at the 0.1% level; these are the three most decisive comparisons in the table, all with raw and adjusted p-values of effectively zero. Within the non-ChatGPT group, the picture is more nuanced: Claude differs from Llama at the 1% level (FDR-adjusted p = 0.007), while the Claude vs DeepSeek and DeepSeek vs Llama comparisons are not significant after FDR correction (p = 0.223 and p = 0.052 respectively). This suggests that Claude, DeepSeek, and Llama form a relatively homogeneous cluster of more sycophantic models, clearly separated from ChatGPT.

The chi-square test on sycophancy intensity categories further confirms the model-level differentiation:

*χ²(6) = 1176.577, p \< 0.001, Cramér's V = 0.158*

A Cramér's V of 0.158 indicates a meaningful association between predicted model and sycophancy intensity. Examination of the standardised residuals illuminates the pattern further. ChatGPT has a very large positive residual for the no-sycophancy category (+12.54) and large negative residuals for the one-keyword (−7.29) and two-or-more-keywords (−16.18) categories: it is far more concentrated in the zero-sycophancy bin than would be expected under independence. Claude and DeepSeek show the mirror image, with strong positive residuals in the higher-intensity categories. Llama also has a large positive residual for the two-or-more-keywords category (+10.16), consistent with its position as the most sycophancy-intensive model in the descriptive statistics.

| **Predicted Model** | **No Sycophancy** | **One Keyword** | **Two or More Keywords** |
|---------------------|-------------------|-----------------|--------------------------|
| ChatGPT             | 9,385             | 2,962           | 1,236                    |
| Claude              | 1,981             | 1,289           | 773                      |
| DeepSeek            | 2,384             | 1,321           | 1,084                    |
| Llama               | 512               | 284             | 281                      |

*Table 10. Sycophancy intensity category counts by predicted model.*

Finally, the permutation test provides a model-free robustness check. Keeping the observed sycophancy keyword counts fixed and randomly permuting model labels 1,000 times, the observed between-model rank variance of 1,657,161.14 yields a permutation p-value of 0.001. The observed group separation is therefore overwhelmingly unlikely to arise by chance. This confirms that the Kruskal-Wallis and chi-square results are not driven by the specific distributional assumptions of those tests.

Taken together, the evidence from this analysis is unambiguous: sycophancy differs significantly across predicted AI models when the full dataset is used. ChatGPT-attributed comments are systematically and substantially less sycophantic than those attributed to Claude, DeepSeek, and Llama. This result is robust across all four testing approaches — Kruskal-Wallis, pairwise Mann-Whitney, chi-square with standardised residuals, and permutation testing — and cannot be attributed to sampling variability or insufficient power, given the large sample sizes involved.

### 3.4 Sycophancy and NLI: A Counterintuitive Relationship

The most theoretically important non-parametric analysis concerns the relationship between sycophancy and NLI labels. If sycophancy were simply equivalent to semantic agreement with the parent post, one would expect comments classified as entailment to exhibit the highest sycophancy levels. The data consistently contradicts this expectation.

#### 3.4.1 Descriptive Evidence

| **NLI Prediction** | **N**  | **Mean KW Count** | **Median** | **% Any Sycophancy** | **% 2+ Keywords** |
|--------------------|--------|-------------------|------------|----------------------|-------------------|
| Neutral            | 17,292 | 0.654             | 0.0        | 42.36                | 15.50             |
| Contradiction      | 2,827  | 0.547             | 0.0        | 36.40                | 13.09             |
| Entailment         | 3,373  | 0.417             | 0.0        | 26.00                | 9.58              |

*Table 10. Sycophancy descriptive statistics by NLI label.*

Entailment comments have the lowest sycophancy rates: only 26.00% contain any sycophancy keyword, and just 9.58% contain two or more. By contrast, neutral comments — which by definition have no logical relationship to the parent post — are the most sycophantic, with 42.36% containing at least one keyword and 15.50% containing two or more. Contradiction comments occupy an intermediate position. The empirical ordering is:

**Neutral \> Contradiction \> Entailment** (in terms of sycophancy keyword count)

#### 3.4.2 Statistical Tests

The Kruskal-Wallis test strongly rejects equality across NLI labels:

*H = 305.140, p \< 0.001, ε² = 0.0129*

The effect size is modest (ε² = 0.0129), indicating that the association is statistically robust but limited in magnitude: NLI category accounts for approximately 1.3% of rank-based variation in sycophancy. All pairwise Mann-Whitney comparisons are significant after FDR correction:

| **Comparison**              | **Mean (Group 1)** | **Mean (Group 2)** | **FDR-adj. p-value** | **Significant?** |
|-----------------------------|--------------------|--------------------|----------------------|------------------|
| Entailment vs Neutral       | 0.417              | 0.654              | \< 0.001             | Yes              |
| Contradiction vs Entailment | 0.547              | 0.417              | \< 0.001             | Yes              |
| Contradiction vs Neutral    | 0.547              | 0.654              | \< 0.001             | Yes              |

*Table 11. Pairwise Mann-Whitney tests: sycophancy by NLI label.*

The chi-square test on intensity categories confirms the pattern:

*χ²(4) = 327.914, p \< 0.001, Cramér's V = 0.084*

The permutation test yields an observed between-NLI rank variance of 453,937.32 and a permutation p-value of 0.001, confirming that the observed separation is highly unlikely under random assignment.

#### 3.4.3 Spearman Correlations with Continuous NLI Scores

To complement the categorical analysis, we compute Spearman rank correlations between sycophancy_keyword_count and the continuous NLI-related scores. These results reinforce the main finding.

| **Variable**        | **Spearman ρ** | **p-value** | **Interpretation**             |
|---------------------|----------------|-------------|--------------------------------|
| Cosine similarity   | 0.0406         | \< 0.001    | Very weak positive association |
| Entailment score    | −0.0747        | \< 0.001    | Weak negative association      |
| Neutral score       | 0.0834         | \< 0.001    | Weak positive association      |
| Contradiction score | 0.0179         | 0.006       | Very weak positive association |

*Table 12. Spearman rank correlations between sycophancy and continuous NLI-related scores.*

The entailment score is negatively associated with sycophancy (ρ = −0.0747), while the neutral score is positively associated (ρ = 0.0834). All correlations are weak in magnitude, confirming that NLI scores are not strong predictors of sycophantic language. The negative sign on the entailment correlation is particularly informative: comments with higher entailment scores — those that most logically follow from the parent post — actually tend to have lower sycophancy keyword counts. This is the opposite of what a simple "sycophancy as agreement" theory would predict.

#### 3.4.4 Generative Validation: LLM-as-a-Judge Stance Detection

To assess the validity of the NLI classification approach, we implemented a complementary generative validation procedure using a large language model as a judge (LLM-as-a-judge). A subset of comment–post pairs was re-evaluated for stance detection by prompting an LLM to assign one of the three canonical labels—entailment, contradiction, or neutral—based on the semantic relationship between the comment and its parent post. The results of this procedure are broadly consistent with the NLI classifications: the LLM-as-a-judge does not systematically contradict the NLI labels, and the two methods converge in the large majority of cases.

However, a systematic divergence emerges in cases where the two methods disagree: when NLI classifies a comment as neutral, the LLM-as-a-judge tends to reclassify it as entailment. This pattern is interpretively important. It suggests that the LLM-as-a-judge applies a broader and more permissive standard of entailment—one that encompasses comments which are thematically consistent or contextually aligned with the parent post, even when they do not strictly follow from it in the logical sense that NLI requires. Conversely, when the LLM judge classifies a comment as entailment, the NLI model is more likely to assign a neutral label, reflecting its comparatively stricter criterion of logical or semantic dependency.

This divergence reflects a genuine conceptual difference between the two validation approaches rather than an error on the part of either method. NLI, as implemented in standard transformer-based models, operationalises entailment as a strict logical or semantic dependency between premise and hypothesis; it is a relatively rigid classifier that reserves the entailment label for cases in which the comment’s content is tightly constrained by the parent post. The LLM-as-a-judge, by contrast, draws on broader pragmatic and contextual reasoning when assigning stance labels, and is more willing to classify a contextually responsive or thematically coherent comment as entailment even in the absence of strict logical dependency. In this sense, NLI-as-a-judge is a more conservative and stringent instrument, while LLM-as-a-judge captures a wider notion of alignment that includes pragmatic relevance and contextual responsiveness. The practical implication for interpretation is discussed further in the Limitations section below.

## 4. LLM-as-a-Judge Analysis: Sycophancy in Contextual and Relational Perspective

This chapter presents a complementary and substantially extended analysis of the sycophancy corpus conducted via an LLM-as-a-judge pipeline applied to a structured archive of 164 output files (161 CSV and 3 TXT). The analysis operates at a different level of granularity than the keyword-based framework employed in the preceding sections: rather than counting sycophancy-related lexical items, it leverages a multi-component scoring system that captures contextual alignment, stance towards the parent comment and post, discourse depth, and relational anchoring. The archive examined comprises four main datasets: a full corpus of 24,563 comments, a stance subset (n = 9,253) with post and parent orientation labels, and a rich subset of 1,500 comments with full contextual labels and a final sycophancy score. All results derive exclusively from these archived outputs.

### 4.1 Archive Structure and Sample Coverage

The archive is structurally coherent and analytically tractable. All 161 CSV files parse without encoding or formatting errors. However, the single most important methodological point is that not all labels exist on the same sample: different analytical operations were applied to samples of different sizes, which means that prevalence estimates from the rich subset cannot be generalised to the full corpus.

| **Dataset**                                          | **Rows** | **Columns** | **Role**                               |
|------------------------------------------------------|----------|-------------|----------------------------------------|
| comment_master_cleaned_for_llm_as_judge_analysis.csv | 24,563   | 104         | Full corpus                            |
| stance_post_parent_analysis_dataset.csv              | 9,253    | 106         | Stance subset (post and parent)        |
| rich_1500_llm_context_analysis_dataset.csv           | 1,500    | 104         | Rich subset                            |
| rich_1500_final_sycophancy_analysis_dataset.csv      | 1,500    | 104         | Final rich subset (identical to above) |

*Table A1. Principal datasets in the LLM-as-a-judge archive.*

The coverage of key label families in the full corpus reflects deliberate sampling choices. The post/parent stance labels (llm_stance_post, llm_stance_parent) are available for 37.7% of the 24,563 comments. External reference detection (llm_external_reference_detected) covers 12.3%. The contextual alignment and final sycophancy labels exist for exactly 6.1% of the corpus, i.e. the rich subset of 1,500 cases. These coverage gaps must be borne in mind when interpreting any aggregate prevalence figure.

Two minor redundancies were identified: the two rich_1500 datasets are byte-for-byte identical, and distribution_extraction_method.csv duplicates extraction_method_summary.csv. These redundancies do not affect results but inflate the apparent file count. Within the rich subset, rich_context_type and extracted_context_type are also identical, as are the corresponding confidence variables.

### 4.2 Post vs. Parent Stance: Structural Asymmetry

A fundamental structural asymmetry characterises comment orientation in this corpus. In the stance subset (n = 9,253), support for the original post is far more frequent than support for the immediate parent comment. The share of comments classified as supportive of the post is 78.3%, while the share supportive of the parent is only 37.9%. The association between llm_stance_post and llm_stance_parent is highly significant but reflects this asymmetry: χ²(4) = 508.33, p = 1.06e−108, Cramér’s V = 0.166.

The cross-tabulation of the two stance variables reveals the structure of this relationship more precisely.

| **llm_stance_post** | **Neutral (parent)** | **Oppose (parent)** | **Support (parent)** |
|---------------------|----------------------|---------------------|----------------------|
| Neutral             | 730                  | 31                  | 329                  |
| Oppose              | 763                  | 55                  | 96                   |
| Support             | 4,115                | 53                  | 3,081                |

*Table A2. Cross-tabulation of llm_stance_post and llm_stance_parent (n = 9,253 stance subset).*

Row-percentage analysis shows that among comments classified as supportive of the post, 42.5% also support the parent. Among neutral comments, this drops to 30.2%; among oppositional comments, to only 10.5%. The standardised residuals identify three particularly informative cells: Support/Support (+6.38), Oppose/Support (−13.45), and Oppose/Neutral (+8.88). Comments that oppose the post are substantially less likely to support the parent than would be expected by chance, while mutual support for post and parent co-occurs above chance expectation.

This asymmetry is substantively important for the sycophancy analysis: the LLM judge detects a tendency for comments to align with the post’s topic or framing far more often than with the conversational move immediately preceding them. As the remaining sections show, it is the latter form of alignment—parent-oriented rather than post-oriented—that drives the sycophancy signal.

### 4.3 Context Extraction: Method and Coverage

In the full corpus, the pipeline extracts a contextual reference for each comment using a heuristic fallback procedure. The extracted_context_type variable identifies six distinct reference types, of which two dominate: prior_sibling_reference (49.8%) and broader_thread_reference (26.5%). Direct reference to the post accounts for 12.6%, while parent_reply covers 9.3%.

| **Context Type**         | **n**  | **%** | **Mean Confidence** |
|--------------------------|--------|-------|---------------------|
| prior_sibling_reference  | 12,242 | 49.8% | 0.858               |
| broader_thread_reference | 6,513  | 26.5% | 0.782               |
| post_direct              | 3,093  | 12.6% | 0.835               |
| parent_reply             | 2,294  | 9.3%  | 0.955               |
| topic_shift              | 227    | 0.9%  | 0.799               |
| ancestor_chain           | 151    | 0.6%  | 0.822               |
| ambiguous                | 43     | 0.2%  | 0.269               |

*Table A3. Distribution of extracted_context_type in the full corpus (n = 24,563).*

Two methodological observations follow from this distribution. First, parent_reply shows the highest mean confidence (0.955), while ambiguous shows the lowest (0.269): the parser distinguishes well-anchored local references from genuinely ambiguous ones. Second, and critically, extraction_method equals heuristic_fallback for 100% of cases. There is no second parsing family in the archive. This means that all subsequent contextual analyses describe the behaviour of this specific heuristic pipeline, not a generalised inference procedure, and cross-pipeline comparison is not possible from these data alone.

### 4.4 Rich Subset: Context, Stance, and Alignment

In the rich subset (n = 1,500), the distribution of context types shifts substantially relative to the full corpus: broader_thread_reference becomes dominant (61.3%), followed by prior_sibling_reference (27.5%), with parent_reply (6.5%) and post_direct (4.0%) accounting for most of the remainder. The LLM judge’s assessment of comment stance towards the extracted context shows a strong tendency towards Entailment (71.9%), with Neutral at 25.3% and Contradiction at only 2.9%. Context alignment classification yields: context_aligned (n = 929), context_neutral_or_new_information (n = 467), and context_contradicting (n = 104).

The association between rich_context_type and llm_stance_context is statistically significant but modest: χ²(10) = 42.10, p = 7.19e−06, V = 0.118. Informative standardised residuals include parent_reply × Contradiction (+3.73), post_direct × Neutral (+2.78), and prior_sibling_reference × Entailment (+1.04): parent-directed replies are more likely to be contradictory, while direct post references are more likely to be judged neutral. By contrast, context_alignment_label and llm_stance_context are not significantly associated (χ²(4) = 4.30, p = 0.366), indicating that alignment and entailment, while conceptually related, capture empirically distinct properties in this workflow.

### 4.5 Final Sycophancy Label and Score Distribution

The LLM-as-a-judge pipeline yields a three-category final label and a continuous likelihood score. The label distribution in the rich subset is deliberately cautious: the majority of cases receive no_clear_sycophancy_signal (n = 782, 52.1%), nearly half receive weak_sycophancy_signal (n = 686, 45.7%), and only 32 cases (2.1%) are classified as moderate_sycophancy_signal. There is no strong sycophancy category in the current schema.

The continuous score final_sycophancy_likelihood has the following descriptive profile.

| **Statistic**      | **Value** |
|--------------------|-----------|
| Mean               | 0.193     |
| Median             | 0.192     |
| Standard deviation | 0.113     |
| Minimum            | 0.000     |
| Maximum            | 0.525     |

*Table A4. Descriptive statistics of final_sycophancy_likelihood (rich subset, n = 1,500).*

The score increases monotonically with label: the mean is 0.103 for no_clear, 0.284 for weak, and 0.436 for moderate cases. The pipeline is therefore internally consistent: the categorical threshold captures genuine gradient differences in the continuous score.

Examination of the component scores that feed into the final likelihood reveals that the pipeline does not weight verbal flattery heavily. Mean values for praise-oriented components are low (syco_praise_score = 0.018, syco_deference_score = 0.010), while structural alignment components are substantially higher (syco_context_alignment_component = 0.666, syco_parent_child_alignment_component = 0.515). Spearman correlations with the final score confirm this: syco_cross_cluster_interaction_component (r = 0.856), syco_parent_child_alignment_component (r = 0.784), and syco_context_alignment_component (r = 0.713) are the strongest predictors. The verbal challenge score is negatively correlated (r = −0.237), consistent with the interpretation that substantive disagreement suppresses the sycophancy signal.

### 4.6 Bivariate Associations with Final Sycophancy Label

Chi-square tests confirm that several discourse-level variables are strongly associated with the final sycophancy label, while others are not. The results are reported in the appendix table; here we focus on the most interpretively significant patterns.

Context alignment is the strongest single predictor of sycophancy (V = 0.434, p = 5.96e−121). Context_aligned comments have mean scores substantially above context_neutral_or_new_information comments, which in turn are above context_contradicting cases. All three pairwise contrasts are significant after FDR correction. This effect is partly structural: the final sycophancy score incorporates a context alignment component directly. However, the direction and magnitude of the association confirm that the pipeline is detecting a coherent construct, not an artefact.

Parent stance is a strong independent predictor (V = 0.235, p = 1.12e−34). Comments classified as supporting the parent have significantly higher scores than neutral-parent comments (Support \> Neutral, FDR-significant); Oppose and Support are not significantly different from each other after FDR correction, which suggests that the sycophancy signal is primarily driven by the presence of supportive alignment, not by the contrast between opposition and neutrality.

Context type predicts sycophancy (V = 0.230), with broader_thread_reference and parent_reply associated with higher weak-sycophancy rates (68.0% and 54.1% respectively) compared to post_direct (16.7%) and prior_sibling_reference (25.7%). The pattern implies that sycophancy emerges preferentially in contexts where the comment is anchored to the broader thread or the immediately preceding message rather than to the original post.

Tone is a significant predictor (V = 0.198): agreeing-tone comments have the highest mean scores (0.277), while disagreeing comments have the lowest (0.162). In FDR-corrected pairwise contrasts, agreeing is significantly higher than disagreeing, neutral_or_unclear, and critical_disagreeing, while disagreeing is lower than both neutral_or_unclear and negative. The tone gradient is monotone in the expected direction.

By contrast, post stance (llm_stance_post) shows no significant association with the final sycophancy label (χ²(4) = 3.60, p = 0.464, V = 0.035). This null result is one of the most substantively important findings of the entire analysis: simple supportive alignment with the original post does not explain sycophancy. It is the relationship with the immediate conversational partner and context—not the topic of the thread—that drives the signal.

### 4.7 Kruskal–Wallis Tests on the Continuous Score

Kruskal–Wallis tests on the continuous final_sycophancy_likelihood replicate the categorical pattern with high precision. The strongest effect is again for context_alignment_label (H = 724.25, p = 5.39e−158, ε² = 0.482). The epsilon-squared value of 0.482 indicates that context alignment alone accounts for nearly half of the rank-based variation in sycophancy scores—a substantially larger effect than any covariate in the keyword-based analysis. The next strongest predictors are llm_stance_parent (H = 128.22, ε² = 0.084), rich_context_type (H = 112.99, ε² = 0.072), and tone_label (H = 83.15, ε² = 0.050).

Variables with smaller but statistically significant effects include llm_external_reference_detected (ε² = 0.008), bert_predicted_model (ε² = 0.007), dominant_topic (ε² = 0.009), and llm_stance_context (ε² = 0.005). As in the categorical analysis, llm_stance_post does not reach significance (H = 4.08, p = 0.130, ε² = 0.001). The model label effect (ε² = 0.007) is noteworthy in that it replicates, at smaller magnitude, the model-level sycophancy differences established in Chapter 3 using keyword counts. Within the rich subset, claude-attributed comments tend to score higher on the continuous likelihood scale than chatgpt-attributed comments, consistent with the earlier finding.

### 4.8 Multivariate Models

#### 4.8.1 OLS Regression on the Continuous Score

An OLS regression on final_sycophancy_likelihood using 1,360 complete cases yields an R² = 0.711, indicating that the set of predictors jointly explains approximately 71% of the variance in the continuous sycophancy score. The model converges without issues.

The most important positive predictors, relative to their respective reference categories, are context_grounding_score (+0.146, p ≈ 0), sycophancy_keyword_count (+0.024, p \< 1e−29), extracted_context_confidence (+0.292, p = 0.005), llm_stance_parent = Support (+0.036, p \< 0.001), comment_depth (+0.010, p = 0.016), and bert_predicted_model = llama (+0.026, p = 0.008). The most important negative predictors are the disagree-family of tone labels (disagreeing: −0.084, critical_disagreeing: −0.076), context types with lower anchoring strength (post_direct: −0.051, prior_sibling_reference: −0.031), and neutral_or_unclear tone (−0.036).

The OLS results reinforce the theoretical picture: sycophancy in this pipeline reflects relational anchoring, depth in the conversation, and lexical alignment with the immediate context, not merely the direction of stance relative to the post. The llama coefficient replicates the model-level direction found in earlier analyses.

#### 4.8.2 Logistic Regression Results

A full logistic regression on the binary sycophancy_any variable does not converge. The non-convergence is traceable to a specific source: the tone category polite_positive appears in only 2 observations, both of which fall in the sycophancy group. This produces near-perfect separation, inflating the coefficient for that category to an astronomically large value. The full logit is therefore not a reliable basis for quantitative inference, though the direction of its coefficients is broadly consistent with the OLS and descriptive evidence.

A reduced logistic model including only support_post, support_parent, their interaction, comment_depth, and sycophancy_keyword_count converges successfully. The key finding is an interaction effect: support_post alone has a negative coefficient (OR = 0.569, p = 0.039), support_parent alone is non-significant (OR = 0.839, p = 0.726), but the joint support_both term is positive and significant (OR = 2.831, p = 0.040). Supporting both post and parent simultaneously more than doubles the odds of a positive sycophancy signal. Depth (OR = 1.767, p = 0.0003) and keyword count (OR = 1.188, p = 0.0017) also increase the odds. The interaction finding is theoretically coherent: it identifies simultaneous relational alignment as the operative mechanism, not unidirectional agreement with either the post or the parent considered separately.

### 4.9 External References and Conversational Memory

In the rich subset, llm_external_reference_detected is fully observed and shows a small but significant association with final sycophancy (χ²(4) = 23.07, p = 0.00012, V = 0.088). The unclear category shows the highest mean score (0.206), above Yes (0.187) and No (0.169). Comments where the model “senses” an external reference but cannot classify it tend to have marginally higher sycophancy scores, suggesting a link between referential ambiguity and accommodating communicative posture. The llm_external_reference_type variable shows an even smaller effect (V = 0.058, p = 0.047) that does not survive FDR correction in the global test set. These external reference results are best treated as secondary and exploratory.

### 4.10 Methodological Limitations of the LLM-as-a-Judge Pipeline

Several limitations constrain interpretation of the results presented in this chapter.

First, the non-comparability of samples across analytical blocks is the most consequential structural limitation. The final sycophancy labels exist for only 6.1% of the full corpus. Prevalence figures from the rich subset (e.g. 2.1% moderate signal) cannot be extrapolated to the 24,563-comment corpus without strong and unverifiable assumptions about the representativeness of the sampling procedure.

Second, the sparsity of the moderate_sycophancy_signal class creates analytical fragility. With only 32 cases, many contingency tables have expected cell counts below 5, and standard chi-square inference is unreliable for cells involving that class specifically. The class is also too structurally homogeneous for logistic regression: 32/32 cases have llm_stance_parent = Support, 32/32 have context_alignment_label = context_aligned, and 31/32 have llm_stance_context = Entailment. The logit on sycophancy_moderate therefore fails by perfect separation and is not available in the archive.

Third, several of the strongest predictors in the OLS model (context_grounding_score, context_alignment_component) are themselves derived from the same pipeline that produces the outcome variable. The high R² = 0.711 partially reflects this structural circularity; it should not be interpreted as evidence of predictive power in a held-out sample.

Fourth, all context extraction uses heuristic_fallback as the sole method. There is no second-pass LLM-based parsing family in the archive. Results therefore describe the behaviour of this specific heuristic under these specific corpus conditions.

### 4.11 Summary: What LLM-as-a-Judge Adds

The LLM-as-a-judge analysis adds three substantive contributions to the account developed in Chapters 1–4.

First, it confirms and substantially deepens the finding that sycophancy is not semantic agreement with the original post. Post stance (llm_stance_post) has no significant association with the final sycophancy score in any test. This replicates, by an entirely different measurement instrument, the core NLI finding that entailment and sycophancy diverge.

Second, it identifies parent alignment and contextual anchoring as the operative mechanisms behind sycophancy. The strongest predictors across all tests are context_alignment_label (ε² = 0.482 in Kruskal–Wallis) and llm_stance_parent (V = 0.235 in chi-square). The reduced logistic model’s interaction term—support_both significantly positive while support_post alone is non-significant or negative—sharpens this: sycophancy is most strongly associated with simultaneous alignment with both post and parent, not with either considered separately.

Third, the pipeline characterises sycophancy as a cautious and structurally grounded signal: only 2.1% of rich-subset comments reach moderate status, and the score is dominated by structural alignment components rather than verbal flattery markers. This complements the keyword-based approach by providing a relational and contextual dimension that pure lexical counting cannot capture. Together, the two approaches converge on a coherent theoretical account: sycophancy in this corpus is a phenomenon of local conversational accommodation, shaped by depth, context type, and alignment with the immediate interactional environment rather than by abstract agreement with the topic of discussion.

### 4.12 Appendix: Summary of Principal Statistical Tests

| **Test**                                  | **χ² / H** | **p-value** | **V / ε²** | **Note**          |
|-------------------------------------------|------------|-------------|------------|-------------------|
| context_alignment_label × final_label     | 564.95     | 5.96e−121   | V = 0.434  | Partly structural |
| llm_stance_parent × final_label           | 165.20     | 1.12e−34    | V = 0.235  | Strong            |
| extracted_context_type × final_label      | 159.05     | 5.10e−29    | V = 0.230  | Sparse cells      |
| tone_label × final_label                  | 117.61     | 1.19e−16    | V = 0.198  | Sparse cells      |
| rich_context_type × llm_stance_context    | 42.10      | 7.19e−06    | V = 0.118  | Modest            |
| external_reference_detected × final_label | 23.07      | 1.23e−04    | V = 0.088  | Small             |
| llm_stance_context × final_label          | 17.03      | 1.91e−03    | V = 0.075  | Small             |
| bert_predicted_model × final_label        | 14.87      | 2.13e−02    | V = 0.070  | Small             |
| llm_stance_post × final_label             | 3.60       | 0.464       | V = 0.035  | Not significant   |
| context_alignment_label (KW)              | H = 724.25 | 5.39e−158   | ε² = 0.482 | Largest effect    |
| llm_stance_parent (KW)                    | H = 128.22 | 1.43e−28    | ε² = 0.084 |                   |
| extracted_context_type (KW)               | H = 112.99 | 9.54e−23    | ε² = 0.072 |                   |
| tone_label (KW)                           | H = 83.15  | 3.81e−14    | ε² = 0.050 |                   |
| llm_stance_post (KW)                      | H = 4.08   | 0.130       | ε² = 0.001 | Not significant   |

*Table A5. Summary of chi-square and Kruskal–Wallis tests from the LLM-as-a-judge archive (rich subset, n = 1,500 unless otherwise noted).*

## 5. Conclusion: What Influences Sycophancy?

This paper set out to understand the distribution and determinants of sycophantic language in a corpus of online posts and comments from an AI-focused community. Across three distinct analytical dimensions — thematic content, predicted model identity, and semantic NLI relationship — the evidence converges on a coherent picture that has both methodological and substantive implications.

### 5.1 Topic and Thematic Context Are the Dominant Drivers

The most powerful predictor of sycophancy in this corpus is the thematic context of the comment. Both the TF-IDF cluster analysis and the NMF topic analysis demonstrate that sycophantic language is not uniformly distributed across the conversational space: it is systematically concentrated in specific thematic regions.

In the NMF analysis, the Kruskal-Wallis effect size ε² = 0.1035 indicates that topic membership accounts for approximately 10.4% of the rank-based variation in sycophancy keyword counts. Topic 10 — characterised by terms of explicit praise and evaluation — has a sycophancy rate of 87.55%, while topics centred on technical protocols such as gossip synchronisation and LAN authentication have rates below 5%. The chi-square association (Cramér's V = 0.244) confirms a moderate and robust link between topic and sycophancy intensity.

The cluster-level analysis tells the same story with greater granularity. Cluster 0, centred on promotional and speculative coinflip content, is an extreme outlier with a sycophancy score nearly 2.5 times that of the next-ranked cluster. At the opposite extreme, the governance and security cluster (cluster 2) — a more analytical and evaluative domain — shows the lowest sycophancy rates in the dataset. These patterns suggest that sycophancy is shaped by the discursive norms and communicative functions of different topical communities: spaces of encouragement, promotion, and social validation generate more sycophantic language; spaces of technical analysis and critical evaluation generate less.

The strong author specialisation across clusters (with a mean dominant-cluster share of 0.906) implies that these topical differences are not merely stylistic noise introduced by individual-level variation. Authors are embedded in specific communities and tend to communicate according to the norms of those communities. Sycophantic language, on this interpretation, is a community-level phenomenon as much as an individual-level one.

### 5.2 Predicted Model Identity Is a Significant but Secondary Predictor

Unlike earlier analyses that were constrained to a small labelled subset of roughly 1,900 comments and found no significant model-level differences, the full-corpus analysis presented here — covering 23,492 observations — reveals a clear and robust association between predicted model identity and sycophancy. The Kruskal-Wallis effect size of ε² = 0.0473 indicates that model identity accounts for approximately 4.7% of the rank-based variation in sycophancy keyword counts, a substantively meaningful quantity. For comparison, this is roughly half the effect size of NMF topic membership (ε² = 0.1035), placing model identity as a genuine but secondary predictor of sycophancy.

The key structural finding is a clear two-tier separation. ChatGPT-attributed comments are dramatically less sycophantic than those attributed to Claude, DeepSeek, and Llama. The mean keyword count for ChatGPT is 0.437 — less than half that of any other model — and the median is 0, meaning the majority of ChatGPT-attributed comments contain no sycophancy keywords at all. The share of comments with two or more keywords is only 9.10% for ChatGPT, compared to 19.12% for Claude, 22.64% for DeepSeek, and 26.09% for Llama. All three pairwise comparisons between ChatGPT and the other models are significant at p \< 0.001. Within the non-ChatGPT group, Claude differs from Llama at p = 0.007, but DeepSeek vs Llama and Claude vs DeepSeek are not significant after FDR correction. Claude, DeepSeek, and Llama therefore form a relatively homogeneous cluster of more sycophancy-intensive models.

These results raise an important interpretive question: what drives the ChatGPT-vs-others separation? Several explanations are possible and are not mutually exclusive. One is that the underlying models genuinely differ in their training-induced propensity for sycophantic language — for instance, if ChatGPT was subjected to RLHF procedures that more aggressively penalised flattery or validation language relative to the other models. A second explanation concerns the classifier itself: the model label is a prediction, not an observed ground truth, and the classification algorithm may systematically attribute certain stylistic features — including lower keyword density — to ChatGPT more often, independently of the actual generating model. A third possibility is a composition effect: if ChatGPT comments are predominantly concentrated in lower-sycophancy topic areas, the model-level difference might partly reflect topical confounding. Disentangling these explanations would require either known model labels or a within-topic analysis of model-level sycophancy, which we leave for future work.

Regardless of the causal mechanism, the empirical finding is robust and consequential: the predicted AI model label is a meaningful correlate of sycophancy in this corpus. The earlier null result — obtained from only 1,934 observations, with very small and roughly balanced model groups — was almost certainly a case of insufficient statistical power. With 23,492 observations and prediction confidence averaging above 0.90, the current analysis is far better positioned to detect genuine model-level differences, and it does so unambiguously.

### 5.3 Sycophancy is Not Semantic Agreement: The NLI Evidence

The most theoretically significant finding of this paper is the relationship between sycophancy and NLI labels. If sycophancy were simply a conversational expression of semantic agreement — the linguistic form taken by the logical content of entailment — then one would expect entailment comments to be the most sycophantic. The data consistently and strongly contradicts this prediction.

Entailment comments are the least sycophantic in the corpus: only 26.00% contain any sycophancy keyword, compared to 36.40% of contradiction comments and 42.36% of neutral comments. The Spearman correlation between the continuous entailment score and sycophancy keyword count is negative (ρ = −0.0747), confirming that higher entailment is associated with lower sycophancy. The empirical ordering — Neutral \> Contradiction \> Entailment — holds robustly across all statistical tests.

This finding invites a reinterpretation of what sycophancy is. If it is not semantic agreement, what is it? The evidence suggests that sycophancy operates primarily as a pragmatic and social communicative function rather than as a semantic one. Sycophantic expressions such as "great," "exactly," "right," and "interesting" function as social lubricants — signals of engagement, approval, and positive reinforcement — that do not necessarily carry or add logical content relative to the parent post. They are most common in comments that are semantically neutral, i.e., comments that do not logically extend or contradict the parent post but instead offer a social response to it. This is consistent with the observation that neutral comments have the highest mean cosine similarity (0.4611), suggesting that they mirror the vocabulary of the parent post without entailing its content — a pattern characteristic of phatic or validating discourse.

Contradiction comments, while less sycophantic than neutral comments, are still more sycophantic than entailment comments. This seemingly paradoxical pattern can be understood if some contradiction comments begin with a validating phrase before proceeding to disagree — a rhetorical strategy of the form "great point, but..." that softens opposition with a sycophantic opener. Such strategies would inflate the keyword count even in logically contradictory comments.

The weak but statistically significant association between NLI category and sycophancy (ε² = 0.0129, Cramér's V = 0.084) confirms that the two constructs are not independent. But the direction of the association and the small effect sizes imply that sycophancy is better understood as a pragmatic style of social validation than as a proxy for semantic alignment.

### 5.4 Summary of Findings

Taken together, the findings of this study lead to three principal conclusions about the determinants of sycophancy in this corpus:

3.  **Topic and thematic context are the primary drivers of sycophancy.** Comments in communities centred on encouragement, praise, and social validation contain far more sycophantic language than comments in technical, analytical, or critical communities. The Kruskal-Wallis effect size for NMF topics (ε² = 0.1035) is the largest of all three covariates examined.

4.  **Predicted model identity is a significant secondary predictor.** When the full corpus of 23,492 observations is used, ChatGPT-attributed comments are substantially and significantly less sycophantic than those attributed to Claude, DeepSeek, and Llama. The effect size (ε² = 0.0473) is meaningful and the result is robust across all four testing approaches. This finding reverses the null result obtained from an earlier, underpowered subset of 1,934 labelled observations.

5.  **Sycophancy is not semantic agreement and cannot be reduced to NLI entailment.** Entailment comments are the least sycophantic, while semantically neutral comments are the most sycophantic. The effect size at the NLI level (ε² = 0.0129) is the smallest of the three covariates. Sycophancy is better understood as a pragmatic style of social validation — a communicative stance of positive reinforcement — than as a direct expression of logical or semantic alignment with the parent post.

### 5.5 Limitations and Future Directions

Several limitations of this study should be acknowledged. First, the keyword-based operationalisation of sycophancy is a pragmatic proxy, not a gold-standard psychological measure. Terms such as "right" and "interesting" can appear in substantive non-sycophantic contexts, and some genuinely sycophantic comments may use vocabulary not captured by the keyword list. Future work should complement keyword counting with more sophisticated sentiment or pragmatic analysis methods, ideally validated against human judgements.

Second, the predicted model labels are outputs of a classification algorithm rather than observed ground truth. Although average prediction confidence is high (above 0.90 for all models), some attributions may be incorrect, and misclassification noise would generally bias any estimated model-level differences toward zero. The strong and consistent ChatGPT effect observed here is therefore unlikely to be an artefact of noisy labels; if anything, it is a lower bound on the true separation. Nevertheless, the question of what drives this separation — genuine model differences, classifier biases, or compositional confounding by topic — cannot be resolved without known ground-truth labels or a within-topic model-level analysis.

Third, the causal direction of the associations identified here cannot be established from observational data. Topic membership predicts sycophancy in a statistical sense, but whether this reflects a causal effect of topic norms on individual communication style — or whether sycophantic authors self-select into certain topic communities — cannot be resolved without longitudinal or experimental data.

Fourth, the NLI-as-a-judge approach employed in this study has inherent limitations that are illuminated by comparison with the generative LLM-as-a-judge validation described in Section 3.4.4. As shown there, the principal source of disagreement between the two methods is concentrated at the neutral–entailment boundary: cases that NLI labels as neutral are frequently reclassified as entailment by the LLM judge. This asymmetry reflects the structural rigidity of NLI-based classification. Standard NLI models are trained to apply a strict logical criterion of entailment and tend to assign a neutral label whenever the comment does not unambiguously follow from the parent post as a logical consequence. The LLM judge, by contrast, applies a more contextually sensitive and pragmatically flexible standard, treating topical alignment and conversational responsiveness as sufficient grounds for an entailment classification even in the absence of strict logical dependency. The practical consequence is that NLI-based labelling likely over-assigns neutral labels relative to a more pragmatically grounded annotation scheme, and may thereby underestimate the extent to which sycophantic comments are semantically aligned with their parent posts in a broad sense. Researchers relying on NLI for stance detection in discourse-analytic settings should be aware of this tendency toward conservative classification and consider augmenting NLI labels with generative validation, particularly when the distinction between neutral and weak entailment is theoretically relevant.

Despite these limitations, the study offers a methodologically rigorous and substantively informative account of how sycophancy is distributed across discourse structure in a naturalistic AI-focused online community. The convergence of evidence across multiple non-parametric methods — Kruskal-Wallis, Mann-Whitney, chi-square, Spearman correlations, and permutation tests — lends confidence to the central conclusions. Sycophancy is not random, not uniformly distributed across models, not equivalent to semantic agreement, and not uniformly distributed across the conversational landscape. It is, above all, a phenomenon of context: shaped by the topics we discuss, the communities we participate in, the models that generate (or are predicted to generate) the text, and the communicative functions those communities assign to language.
