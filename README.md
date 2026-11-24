# Steam Game Recommendation: Playtime-Aware BPR

CSE 258 Assignment 2. Does playtime-weighted BPR outperform standard BPR on top-K Steam game recommendation? See [doc.md](doc.md) for the full plan, division of work and schedule.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install        # strips notebook outputs on commit; run once per clone
python -m ipykernel install --user --name steam-bpr
jupyter notebook workbook.ipynb
```

Select the `steam-bpr` kernel in Jupyter. The notebook downloads the raw dataset (~74 MB) into `data/` on first run. `data/` is git-ignored.

## Working in the shared notebook

All code lives in `workbook.ipynb`, organized by the five graded sections. Section headers are tagged with an owner: `[S]` Svante, `[T]` Tony, `[S+T]` both.

- Only edit cells in your own sections.
- Pull before you start, and commit and push small changes often.
- Tell each other before editing so you don't both change the notebook at once. Git cannot merge two sets of notebook edits cleanly.
- Section 2.3 saves the split to `data/*.parquet`, so model cells (Section 3 onward) can load it without rerunning preprocessing.

## Interface

- `data/train.parquet`, `data/val.parquet`, `data/test.parquet`: `user_id` (int), `item_id` (int), `playtime_hours` (float), with contiguous IDs.
- `data/meta.json`: `num_users`, `num_items`, `seed`, ID mappings.
- Every model has `score_users(user_ids)` returning a `(len(user_ids), num_items)` NumPy array of scores.
- `evaluate(model, eval_df, mask_dfs, num_items)` in Section 1.1 handles masking, ranking and metrics for all models.

## Submission

```bash
jupyter nbconvert --to html --execute workbook.ipynb   # produces workbook.html
```

Delete the team-notes comment at the top of the notebook before exporting. Submit `workbook.html` and `video_url.txt` (one line: a Google Drive link with "anyone with the link" access, or a YouTube link).
