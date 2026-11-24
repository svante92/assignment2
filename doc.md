# Steam Game Recommendation: Playtime-Aware BPR

**Course:** CSE 258 — Assignment 2  
**Team:** Tony and Svante  
**Timeline:** November 23–30, 2026 (confirm official deadline on Gradescope)

## 1. Project overview

We investigate whether incorporating historical playtime improves personalized game recommendation compared with binary implicit feedback. Given a user's observed Steam library and playtime, the model ranks games outside the observed library. The target is held-out game ownership, not future purchases, ratings, or predicted hours of play.

**Research question:** Does playtime-weighted Bayesian Personalized Ranking (BPR) outperform standard BPR on top-K game recommendation?

**Hypothesis:** Playtime provides a useful preference signal, but heavy-tailed playtime and differences in user activity may make normalization important. Negative results are acceptable and will be analyzed.

## 2. Dataset and preprocessing

- Source: [Steam data, Version 1 — User and Item Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data).
- Required fields: `user_id`, `item_id`, `playtime_forever` (minutes; convert to hours).
- Flatten nested item lists, remove duplicate user-game pairs, validate nonnegative playtime, and retain zero-playtime ownership interactions.
- Iteratively filter users with fewer than 10 games and games with fewer than 5 users. Report counts before and after filtering.
- Use a seeded per-user random 80/10/10 train/validation/test split, ensuring at least one interaction in each partition. Only train-supported items are eligible for evaluation; report excluded held-out interactions and users.
- Fit all model statistics and playtime transformations using training data only. Use validation for hyperparameter selection; keep test data untouched until final evaluation.
- If full-data training is too expensive, use a fixed seeded user subset, report the sampling procedure, and use identical splits for every model.

### Exploratory data analysis

Report user/item/interaction counts, sparsity, zero-playtime fraction, playtime quantiles, and plots of games per user, users per game, and raw/log-transformed playtime. Explain that ownership is imperfect implicit feedback and that the dataset is a snapshot rather than a chronological purchase history.

## 3. Models

All models rank the same candidate catalog for each user.

### 3.1 Popularity baseline

Score each game by its number of distinct training users.

### 3.2 Item-based Jaccard similarity

For games $i$ and $j$, let $U_i$ and $U_j$ be their training users:

$$J(i,j)=\frac{|U_i\cap U_j|}{|U_i\cup U_j|}.$$

Score an unseen candidate by summing its similarities to games in the user's training history. Use sparse matrices and optionally retain the top 100 neighbors per item for efficiency; document the setting.

### 3.3 Standard BPR

Learn user and item embeddings, with score $s(u,i)=p_u^Tq_i$.

Sample triplets $(u,i,j)$ where $i$ is a training-owned game and $j$ is not in the user's training history. Unobserved games are treated as sampled negatives, not verified dislikes. Do not use validation/test ownership to construct training negatives.

$$\mathcal L=\mathbb E[-\log\sigma(s(u,i)-s(u,j))]+\lambda\|\Theta\|_2^2.$$

### 3.4 Playtime-weighted BPR (proposed)

$$\mathcal L=\mathbb E[-w_{ui}\log\sigma(s(u,i)-s(u,j))]+\lambda\|\Theta\|_2^2.$$

Let $t_{ui}$ denote training playtime in hours and $a_{ui}=(1+\log(1+t_{ui}))^\alpha$.

| Variant | Raw weight |
|---|---|
| Standard BPR | 1 |
| Log-weighted BPR | $a_{ui}$, with $\alpha=1$ initially |
| User-normalized BPR | $a_{ui}/\operatorname{mean}_{k\in I_u^{train}}a_{uk}$ |

Normalize weights to have mean 1 under the training triplet sampling distribution (use uniform training-interaction sampling). The same architecture, sampler, optimizer, embedding size, and regularization should be used across BPR variants for the controlled comparison. Use a numerically stable `softplus(s_neg - s_pos)` loss.

## 4. Training and evaluation protocol

- Suggested starting configuration: 64-dimensional embeddings, Adam, learning rate 0.001, batch size 4096, up to 30 epochs, early stopping on validation NDCG@10. Tune a small shared configuration if needed.
- Sample one or more negatives per positive using the same policy for every BPR variant.
- Rank all train-supported games in batches. For validation, mask training games; for test, mask training and validation games. Do not retrain on validation unless the protocol is explicitly changed for all models.
- Compute macro-averaged Recall@10, Recall@20, NDCG@10, and AUC over eligible users. Treat held-out ownership as relevance; unobserved candidates are not confirmed negatives.
- Use deterministic tie handling, fixed seeds, and identical candidate sets across models. If full-catalog evaluation is infeasible, use the same fixed sampled candidates for every model and disclose that metrics are sampled.
- Use validation NDCG@10 as the primary model-selection metric. If compute permits, repeat BPR experiments with three seeds and report mean and standard deviation.

## 5. Experiments

### Experiment A — Main comparison

| Model | Recall@10 | Recall@20 | NDCG@10 | AUC |
|---|---|---|---|---|
| Popularity | TBD | TBD | TBD | TBD |
| Jaccard | TBD | TBD | TBD | TBD |
| Standard BPR | TBD | TBD | TBD | TBD |
| Log-weighted BPR | TBD | TBD | TBD | TBD |
| User-normalized BPR | TBD | TBD | TBD | TBD |

### Experiment B — Playtime sensitivity

Evaluate $\alpha\in\{0,0.25,0.5,1,2\}$ on validation data for the log-weighted variant; $\alpha=0$ is standard BPR. Plot validation NDCG@10 versus $\alpha$. Select the best value using validation only and report its final test result separately if it differs from the fixed log-weighted model. Reduce to $\{0,0.5,1,2\}$ if compute is limited.

### Experiment C — User activity analysis

Divide users by total training playtime into bottom 25%, middle 50%, and top 25%. Compare standard BPR and playtime-weighted models on each group; report group sizes. Discuss whether observed differences might be explained by library size or popularity bias rather than playtime alone.

Optional, only after core experiments: catalog coverage, popularity bias, or paired bootstrap confidence intervals.

## 6. Division of work

| Component | Owner | Deliverable |
|---|---|---|
| Dataset download, preprocessing, split | Svante | Processed data, split files, dataset statistics |
| EDA | Svante | Figures and data discussion |
| Popularity and Jaccard | Svante | Baseline implementation and scores |
| Evaluation pipeline | Svante | Shared ranking metrics and candidate masking |
| Standard BPR | Tony | PyTorch model, sampler, training loop |
| Weighted BPR variants | Tony | Weight transformations and training configuration |
| Hyperparameter/alpha experiments | Tony | Training logs and ablation results |
| Main comparison and user subgroup analysis | Both | Final tables, plots, interpretation |
| Related work | Both | Svante: Steam and implicit feedback; Tony: BPR and weighting |
| Final notebook and presentation | Both | Integrated notebook, slides, recorded video |

### Integration contract

Svante owns preprocessing and evaluation; Tony owns the trainable recommendation models. All code lives in Jupyter notebooks; there are no separate Python modules. Agree on this interface before implementing independently:

- `train.parquet`, `val.parquet`, `test.parquet`: integer `user_id`, integer `item_id`, float `playtime_hours`; use shared contiguous ID mappings and persist metadata (`num_users`, `num_items`, seed). Saving the split lets BPR cells load data without rerunning preprocessing.
- Baselines and BPR expose `score_users(user_ids)` returning scores in shared item-ID order, in batches as needed.
- Evaluation handles masking, eligibility, ranking metrics, and activity groups centrally.
- Save results as CSV or JSON with model, seed, split, metrics, and configuration. Do not commit raw datasets or large checkpoints.
- Both members work in the single shared `workbook.ipynb`, ordered by the five graded sections. Each owns their own sections and does not edit the other's cells.
- Git cannot merge notebooks reliably, so avoid conflicts: pull before starting, commit and push small changes often, and tell each other before editing. Clear outputs before committing (or use `nbstripout`) to keep diffs small; run the notebook fully only for the final export.
- Peer graders only see the exported `workbook.html`, so every piece of code shown in the presentation must appear in it. Both review the final results and confirm the notebook runs top to bottom.

## 7. Repository structure

```text
.
├── doc.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # ignored raw/processed datasets
├── results/               # metric CSV/JSON files
├── figures/
├── workbook.ipynb         # the single shared notebook (submitted as workbook.html)
└── workbook.html
```

The final `workbook.ipynb` follows the graded sections: (1) predictive task, (2) data and EDA, (3) models, (4) evaluation and results, (5) related work and limitations.

Recommended stack: Python, NumPy, Pandas, SciPy, PyTorch, Matplotlib, PyArrow, Jupyter. Pin dependencies after verifying the environment.

## 8. One-week schedule

| Date | Tony | Svante | Milestone |
|---|---|---|---|
| Nov 23 | Agree on model interface | Download data, agree on split schema | Finalize plan |
| Nov 24 | Implement BPR and sampler | Preprocess, EDA, popularity | Data and initial baseline |
| Nov 25 | Train standard BPR | Jaccard and shared evaluation | End-to-end pipeline |
| Nov 26 | Weighted BPR experiments | Verify metrics and baseline results | Main comparison |
| Nov 27 | Alpha ablation | User activity analysis | Freeze core results |
| Nov 28 | Modeling and evaluation write-up | EDA, dataset and related work write-up | Integrated notebook/slides |
| Nov 29 | Present modeling/results | Present task/data/baselines | Record and validate submission |
| Nov 30 | Joint final checks | Joint final checks | Buffer; submit by official deadline |

If behind schedule, prioritize the three baselines, log-weighted BPR, and a correct evaluation pipeline; reduce the alpha sweep before sacrificing notebook or presentation quality.

## 9. Related work

- Rendle et al. (2009), [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618).
- Hu, Koren, and Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets*, ICDM.
- Pathak, Gupta, and McAuley (2017), *Generating and Personalizing Bundle Recommendations on Steam*, SIGIR. See the dataset page for the paper and dataset provenance.

Compare our method and findings with these works; do not claim that playtime weighting is a novel research contribution without a broader literature review.

## 10. Final deliverables and checklist

- [ ] Reproducible data preprocessing and shared split
- [ ] EDA plots and dataset statistics
- [ ] Popularity, Jaccard, and standard BPR
- [ ] Playtime-weighted and user-normalized BPR
- [ ] Main comparison, alpha ablation, and user activity analysis
- [ ] Discussion of limitations and related work
- [ ] Clean, executed `workbook.ipynb` exported as `workbook.html`
- [ ] Approximately 20-minute presentation (both members participate)
- [ ] `video_url.txt` containing an accessible video link; verify submission requirements
- [ ] Submit peer grading report by its separate deadline

### Suggested presentation allocation

| Section | Presenter | Minutes |
|---|---|---:|
| Predictive task | Svante | 2 |
| Dataset and EDA | Svante | 4 |
| Models and objectives | Tony | 5 |
| Evaluation and results | Both | 6 |
| Related work and limitations | Both | 3 |

The grading rubric allocates 5 points to each of the five content sections and 4 points to peer grading. Prioritize a clear, correct and reproducible analysis over extra architectures.
