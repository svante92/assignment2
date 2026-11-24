# Steam Game Recommendation: Playtime-Aware BPR

**Course:** CSE 258 — Assignment 2  
**Team:** Tony and Svante  
**Timeline:** November 23–30, 2026 (confirm official deadline on Gradescope)

## 1. Project overview

We investigate whether incorporating historical playtime improves personalized game recommendation compared with binary implicit feedback. Given a user's observed Steam library and playtime, the model ranks games outside the observed library. The target is held-out game ownership, not future purchases, ratings, or predicted hours of play.

**Research question:** Does playtime-weighted Bayesian Personalized Ranking (BPR) outperform standard BPR on top-K game recommendation?

**Hypothesis.** Longer playtime may strengthen the ownership signal. The heavy-tailed playtime distribution and differences in user activity motivated the logarithmic transform and per-user normalization.

**Finding.** Standard BPR led all five models on the full test split, with NDCG@10 of 0.2790. The validation sweep selected equal weights ($\alpha=0$). Both playtime-weighted variants recovered a higher share of held-out games users had played, and a lower share of held-out games with zero recorded playtime.

## 2. Dataset and preprocessing

- Source: [Steam data, Version 1 — User and Item Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data).
- Required fields: `user_id`, `item_id`, `playtime_forever` (minutes; convert to hours).
- Flatten nested item lists, remove duplicate user-game pairs (keeping the larger playtime), validate nonnegative playtime, and retain zero-playtime ownership interactions.
- Remove entries that are not standalone games (test/beta builds Steam grants alongside a game, dedicated servers, SDKs and editors, soundtracks, demos) by name only, with an explicit exceptions list for real games whose names match. This removed 76 entries (3.1% of interactions), including *Dota 2 Test*, the most-owned entry in the raw data.
- Iteratively filter users with fewer than 10 games and games with fewer than 5 users. Report counts before and after filtering. Result: 56,457 users, 9,130 games, 4,876,826 interactions.
- The seeded per-user split produced 3,896,608 training interactions, 490,109 validation interactions, and 490,109 test interactions. All 56,457 users were eligible for validation and test. Every held-out game appeared in training.
- Fit model statistics and playtime transformations on training data. Use validation for hyperparameter and checkpoint selection. Use the full shared split for the reported experiments.

### Exploratory data analysis

Section 2.4 of the notebook reports dataset counts, sparsity, zero-playtime share, playtime quantiles, and plots of library size, game popularity, and raw and log-transformed playtime. The dataset is a snapshot of ownership and recorded playtime, so ownership serves as a proxy for preference.

## 3. Models

All models rank the same candidate catalog for each user.

### 3.1 Popularity baseline

Score each game by its number of distinct training users.

### 3.2 Item-based Jaccard similarity

For games $i$ and $j$, let $U_i$ and $U_j$ be their training users:

$$J(i,j)=\frac{|U_i\cap U_j|}{|U_i\cup U_j|}.$$

Score an unseen candidate by summing its similarities to games in the user's training history. Compute similarities blockwise with sparse matrices. The full validation sweep tested 25, 100, 400, and all neighbors per item. All neighbors gave the highest validation NDCG@10 (0.2335) and fit in memory as a dense 9,130 × 9,130 matrix (about 330 MB). Break score ties by lower item ID. [Jaccard validation results](results/jaccard_validation_sweep.csv)

### 3.3 Standard BPR

Learn user and item embeddings, with score $s(u,i)=p_u^Tq_i$.

Shuffle the eligible training interactions each epoch. For each training-owned game $i$ of user $u$, sample one game $j$ uniformly from the train-supported catalog outside that user's training history. The sampled negative represents missing ownership in the training data.

$$\ell_{uij}=\operatorname{softplus}(s(u,j)-s(u,i)),\qquad R_{uij}=\|p_u\|_2^2+\|q_i\|_2^2+\|q_j\|_2^2.$$

Standard BPR minimizes the mean triplet loss $\mathcal L=\mathbb E[\ell_{uij}+\lambda R_{uij}]$.

### 3.4 Playtime-weighted BPR (proposed)

$$\mathcal L=\mathbb E[w_{ui}\ell_{uij}+\lambda R_{uij}].$$

Let $t_{ui}$ denote training playtime in hours and $a_{ui}=(1+\log(1+t_{ui}))^\alpha$.

| Variant | Raw weight |
|---|---|
| Standard BPR | 1 |
| Log-weighted BPR | $a_{ui}$, with $\alpha=1$ in the main comparison |
| User-normalized BPR | $a_{ui}/\operatorname{mean}_{k\in I_u^{train}}a_{uk}$ |

Normalize weights to have mean one across interactions eligible for triplet sampling. All BPR variants share the architecture, sampler, optimizer, embedding dimension, and regularization.

## 4. Training and evaluation protocol

- BPR uses 64-dimensional embeddings, Adam with learning rate 0.001, batch size 4,096, sampled L2 regularization of 0.0001, and seed 42.
- Train for up to 100 epochs. Stop after five consecutive epochs without a new best validation NDCG@10 and restore the best checkpoint. Evaluate NDCG@10 on all validation users each epoch. The selected checkpoints were epochs 91, 73, and 59 for standard, log-weighted, and user-normalized BPR, respectively. [Checkpoint comparison](results/bpr_extended_validation.csv)
- Rank the full train-supported catalog in batches. Mask training games on validation. Mask both training and validation games on test. Use the same candidates and deterministic tie-breaking for all models.
- Report macro-averaged Recall@10, Recall@20, NDCG@10, and AUC over eligible users. Held-out ownership defines relevance. The reported estimates are from one seed.

## 5. Experiments

### Experiment A — Main comparison

| Model | Recall@10 | Recall@20 | NDCG@10 | AUC |
|---|---|---|---|---|
| Popularity | 0.1480 | 0.2110 | 0.1577 | 0.9500 |
| Jaccard | 0.2422 | 0.3102 | 0.2702 | 0.9611 |
| Standard BPR | **0.2603** | **0.3567** | **0.2790** | **0.9772** |
| Log-weighted BPR | 0.2435 | 0.3382 | 0.2586 | 0.9769 |
| User-normalized BPR | 0.2370 | 0.3305 | 0.2524 | 0.9766 |

The test comparison covers all 56,457 users and uses the validation-selected checkpoints from the extended BPR runs. [Full test results](results/main_comparison.csv)

### Experiment B — Playtime sensitivity

Across $\alpha\in\{0,0.25,0.5,1,2\}$, validation NDCG@10 declined from 0.2364 to 0.2023. The best positive exponent was 0.25, at 0.2329. Validation selected $\alpha=0$, so the selected checkpoint is already represented by standard BPR in the test table. [Sweep results](results/bpr_alpha_sweep.csv) · [Sweep plot](figures/bpr_alpha_sweep.png)

### Experiment C — User activity analysis

Standard BPR led both weighted variants in the bottom quarter, middle half, and top quarter of total training playtime. The log-weighted model's NDCG@10 gap narrowed from 0.0250 in the bottom quarter to 0.0159 in the top quarter. When grouping by training library size, the gap narrowed from 0.0216 to 0.0191. The groupings agree for 57% of users. The comparison is descriptive because playtime and library size are correlated. [Activity breakdown](results/bpr_user_activity_test.csv)

### Played and never-played held-out games

Of the held-out test interactions, 34.6% had zero recorded playtime. At rank 20, log-weighted BPR recovered 29.7% of the played games and 16.3% of the never-played games. Standard BPR recovered 28.1% and 23.7%, respectively. These rates pool held-out interactions within the two playtime groups. [Hit rates by recorded playtime](results/heldout_hits_by_playtime.csv)

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
- Save results as CSV or JSON with model, seed, split, metrics, and configuration. The raw `australian_users_items.json.gz` is committed at the repo root; do not commit processed data or model checkpoints.
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
├── australian_users_items.json.gz  # raw dataset (committed)
├── data/                  # ignored processed data (split parquet files, meta.json)
├── results/               # metric CSV/JSON files
├── figures/
├── workbook.ipynb         # the single shared notebook (submitted as workbook.html)
└── workbook.html
```

The final `workbook.ipynb` follows the graded sections: (1) predictive task, (2) data and EDA, (3) models, (4) evaluation and results, (5) related work and limitations.

Recommended stack: Python, NumPy, Pandas, SciPy, PyTorch, Matplotlib, PyArrow, Jupyter. Pin dependencies after verifying the environment.

## 8. Submission milestones

As of November 24, the modeling, evaluation, and write-up are in the shared notebook. Tony's modeling presentation segment is ready to incorporate into the team deck. [Tony's slide copy and speaker notes](tony_presentation.md)

| Target | Remaining work |
|---|---|
| Nov 28 | Finish slide prep and review the notebook together. Tighten Section 4.5 to describe recorded playtime on held-out games. |
| Nov 29 | Run and verify the notebook export. Record the two presentation blocks and combine them into one video. |
| Nov 30 | Verify both submission files and submit by the deadline posted on Gradescope. |

These are team planning dates. Check Gradescope for the official assignment and peer grading deadlines.

## 9. Related work

- Rendle et al. (2009), [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618).
- Hu, Koren, and Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets*, ICDM.
- Pathak, Gupta, and McAuley (2017), *Generating and Personalizing Bundle Recommendations on Steam*, SIGIR. See the dataset page for the paper and dataset provenance.

Section 5 compares the results with prior work on ranking, confidence weighting, and Steam behavior. [Notebook](workbook.ipynb)

## 10. Final deliverables and checklist

- [x] Reproducible data preprocessing and shared split
- [x] EDA plots and dataset statistics
- [x] Popularity, Jaccard, standard BPR, and both playtime-weighted variants
- [x] Extended training, test comparison, alpha sweep, and subgroup analysis
- [x] Modeling overview, related work, and limitations
- [x] Tony's five-minute slide copy and speaker notes
- [ ] Review Section 4.5 wording and remove team notes from the notebook
- [ ] Run `workbook.ipynb` end to end, export `workbook.html`, and verify it in a browser
- [ ] Finish the team deck, record separate presentation blocks, and combine them into one roughly 20-minute video
- [ ] Verify the shareable video link, put it in `video_url.txt`, and submit both files on Gradescope
- [ ] Submit peer grading report by its separate deadline

### Suggested presentation allocation

Svante records the opening 10 minutes. Tony records the closing 10 minutes. Combine the two recordings in that order for submission.

| Section | Presenter | Minutes |
|---|---|---:|
| Predictive task and evaluation protocol | Svante | 2 |
| Dataset and EDA | Svante | 4 |
| Popularity and Jaccard baselines, plus related work | Svante | 4 |
| BPR models, objectives, and training | Tony | 5 |
| Test results, playtime findings, and limitations | Tony | 5 |

The grading rubric allocates 5 points to each of the five content sections and 4 points to peer grading. Prioritize a clear, correct and reproducible analysis over extra architectures.
