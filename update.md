codex exec --dangerously-bypass-approvals-and-sandbox -C . \
 “Implement “Level 1” co-training (EM-style) so evolution is integrated into classifier training.

Goal:
- Instead of freezing the classifier (Level 0), alternate between:
  (A) training the classifier on subgraph-derived features, and
  (B) running evolutionary search using the current classifier as the fitness oracle to generate better subgraphs,
  then retraining the classifier on those evolved subgraphs.
- This should reduce the distribution mismatch (classifier trained only on k-hop/full-embed but evaluated on evolved subgraphs) and improve both performance + hypothesis diversity.

What to build (high-level):
1) Add a new CLI mode in `scripts/run_experiments.py`:
   - `--cotrain_rounds N` (default 0 for old behavior; Level 0)
   - when N>0 run co-training for N rounds (Level 1).
   - Also add `--train_subgraph_method {khop,random_walk,shortest_paths}` for the initial round (default khop).
   - Add `--evo_train_topk K` (default 1 or 3) = how many evolved subgraphs per TRAIN pair to keep for retraining.
   - Add `--evo_negatives {khop,random_walk,none}` default khop for negatives (don’t evolve negatives initially).
   - Add `--mix_khop_frac f` (default 0.5): fraction of training minibatches drawn from baseline subgraphs vs evolved subgraphs.

2) Co-training loop (per split and per seed):
   Round 0:
   - Build candidate pools for TRAIN/VAL/TEST pairs (cached).
   - Train classifier C0 using baseline subgraphs for TRAIN pairs (khop by default). Evaluate on VAL (baseline + evo optional) to log.
   For round r=1..N:
   - Run evolution on TRAIN positives (and optionally a subset of TRAIN negatives) using classifier C_{r-1} as fitness.
     * For each TRAIN positive pair, store top `evo_train_topk` evolved subgraphs.
     * For TRAIN negatives, either:
       - keep baseline subgraphs (cheap) OR
       - optionally evolve a small subset (if enabled).
   - Create a new training dataset where each (drug,disease,label) has 1 or more associated subgraphs:
     * positives: evolved subgraphs (topK) plus optionally baseline subgraphs
     * negatives: baseline subgraphs (unless evolving negatives is enabled)
     * If multiple subgraphs per pair, treat them as separate rows (data augmentation).
   - Retrain classifier from scratch (recommended) or warm-start (optional flag) to get C_r.
   - Evaluate C_r on VAL and TEST using:
     * baseline subgraphs (khop/random_walk/shortest_paths)
     * evolved subgraphs (run evolution on VAL/TEST queries separately as usual)
   - Save per-round metrics to `results/tables/cotrain_metrics.csv` and per-round artifacts.

3) Fitness function remains:
   Fitness(S) = log(p_positive(d,y,S)+1e-9) - lambda_size * |E(S)|
   (keep existing lambda; optionally add a novelty term later, but implement co-training first).

4) Caching / artifacts:
   - `artifacts/evo_train/{split}/{seed}/round_{r}/...` for evolved TRAIN subgraphs.
   - Keep existing `artifacts/evo/{split}/{seed}/...` for evolved TEST subgraphs.
   - Cache features per (query_id, subgraph_hash) as already designed.

5) Reporting:
   - Update `main_metrics.csv` to include a column `round` and `mode` (baseline vs evolution) so we can compare:
     * Level 0 vs co-trained (Level 1)
     * Round 0 vs Round N
   - Also compute diversity metrics on TEST topK evolved hypotheses each round to show co-training improves diversity (optional but helpful).

Default settings (reasonable for PoC):
- `--cotrain_rounds 2`
- `--evo_train_topk 1` (or 3 if affordable)
- `--mix_khop_frac 0.5`
- Keep budgets/pop/generations as current defaults.

Implementation detail:
- Ensure no leakage: evolution on TRAIN uses only training KG adjacency and candidate pools derived from training KG; evaluation is still on held-out splits.
- For speed: allow `--evo_train_max_cases` to cap how many TRAIN positives get evolved each round (default e.g. 2000), but do not cap TEST evolution too low for publication.

Please implement this end-to-end, update the CLI help text, and add a short README section “Co-training mode (Level 1)” with an example command.”
