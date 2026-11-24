# Models and training for Steam recommendation

Tony’s five-minute segment for the CSE 258 presentation. Start after Svante introduces the task, dataset, and baselines.

## 0:00–0:55 · What the models score

**On the slide**

- Observed ownership and recorded playtime in the training data
- One score per game in the training catalog
- Popularity counts owners. Jaccard measures ownership overlap. BPR learns user and game embeddings.

**Speaker notes**

Thanks, Svante. We give each model the observed part of a user’s library and ask it to score every game in the training catalog. The evaluator removes known games before ranking, then checks whether the model recovered games from the hidden part of the library. That gives all five methods the same recommendation task.

Popularity builds a single ranking by counting owners. Jaccard looks for games that tend to be owned by the same people as games in your library. It builds personalized scores from item similarities. Our third approach, Bayesian Personalized Ranking, learns compact vectors for users and games. Those vectors personalize the scores with about 4.2 million parameters across all users and games.

## 0:55–2:05 · Bayesian Personalized Ranking

**On the slide**

- Score = user embedding · game embedding
- Pair one training-owned game with one sampled game outside the user’s training library
- Minimize the pairwise ranking loss with sampled L2 regularization

**Speaker notes**

BPR gives each user and each game a 64-dimensional vector. Their dot product is the score. To teach the model what a good ordering looks like, we take one game the user owns in training and sample another game from outside that user’s training library. We draw that second game uniformly from the eligible catalog. The available games reflect what we observe in training, so the sampled game can be one the user would enjoy.

The loss is based on the difference between the two scores. It is large when the sampled game scores above the observed game. It shrinks as the observed game pulls ahead. We use softplus to compute that loss stably, and add an L2 penalty on the three participating vectors. Each epoch shuffles the owned interactions and samples one comparison per interaction. This avoids enumerating all possible pairs for every user.

## 2:05–3:25 · Playtime weighting

**On the slide**

| Variant | How each training pair contributes |
|---|---|
| Standard BPR | Equal weight |
| Log-weighted BPR | More weight for longer recorded playtime |
| User-normalized BPR | A mean weight of one within each user’s library |

**Speaker notes**

The main experiment changes how much an owned game contributes to the ranking loss. Standard BPR gives every pair weight one. For the log-weighted version, we convert playtime to hours, take log of one plus hours, add one, and raise the result to alpha. The logarithm compresses the long tail, and the raw weight starts at one for a game with zero recorded hours. Alpha controls how strongly the loss favors games with more hours. At zero, all weights are one.

The user-normalized version also divides by the average weight in that user’s training library. A heavily played game still counts more than a lightly played game for the same user. The user’s average weight returns to one, preserving the total weight their library receives in standard BPR. For both schemes we keep the average weight across eligible training interactions at one. The embeddings, regularizer, and negative sampling rule are shared across all three variants.

## 3:25–5:00 · Training and checkpoint selection

**On the slide**

- Adam, learning rate 0.001, batch size 4,096
- Up to 100 epochs, with patience of five
- Select by validation NDCG@10 over all 56,457 users
- [Validation curves with the selected checkpoints](figures/bpr_extended_validation.png)

**Speaker notes**

We set up the comparison to make the weighting rule the main difference between models. All variants use the same split, seed, optimizer, and training budget. We train in batches with Adam and check NDCG at ten after each epoch on the full validation set. NDCG gives more credit when a held-out game appears closer to the top, which is what we want from a short recommendation list.

Here the vertical line marks the original 30-epoch limit. All three models improved after that point, so we gave them up to 100 epochs. We stop when five consecutive epochs pass without a new best validation score, and we restore the best checkpoint. The dots mark those checkpoints. Standard BPR peaked at epoch 91, the log-weighted model at 73, and the user-normalized model at 59.

We also used the same budget to sweep alpha on validation. The best setting was zero, which is standard BPR. We fixed the selected checkpoints before computing the final test metrics. With the training choices established, let’s look at how the five models performed on the same test users.

## Notebook references

Use Sections 3.1–3.4 for model definitions, Section 4.2 for the alpha sweep, and Section 4.4 for the checkpoint curves. The complete test comparison is in Section 4.1. [Workbook](workbook.ipynb)
