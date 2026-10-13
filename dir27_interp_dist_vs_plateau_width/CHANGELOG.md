# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-09-23 — first full REPORT.md and RESULTS.md

- RESULTS.md: replaced the template with S1–S5 results for the full 50,256-token sweep. Added the flag
  counts, variable percentiles, median width per D range, W_final band examples, and 4 figures.
- REPORT.md: replaced the template with the full report. It answers the three PLAN questions
  (distance, depth, grouping) using Figures 1–3. No earlier numbers were superseded.

## 2026-09-23 — human_feedback.txt: zoomed Figure 1
- Added plots/fig1_zoom_distance_vs_width.png (new `zoom` mode in experiments/s3_plots.py; reads
  results/sweep.csv, so the CSV and the existing figures are byte-identical). x: D in [1.6, 3.3]; y: per-panel
  min–max W of tokens in that range (block 18: 0.065–0.627; block 35: 0.021–0.650); 50,164 tokens.
- REPORT.md: new Figure 2 after Figure 1 in Finding 1; old Figures 2, 3 renumbered to 3, 4.
- RESULTS.md: new Figure 3 after the D-vs-W figure; old Figures 3, 4 renumbered to 4, 5.

## 2026-09-23 — human_feedback_2: representative-D examples and widest-token list
- `plots/fig3_representative_curves.png` panel (a): replaced ` of`, ` the`, ` good`, ` great`, ` large`
  (D 1.35–1.67, W_final 0.046–0.466) with ` domestically`, ` nickel`, ` bulletin`, ` oxide`, ` nutshell`
  (D 2.49–2.54, W_final 0.049–0.476), chosen nearest the min/quartiles/max of W_final among the 8,307 tokens
  with D in 2.45–2.55. Panel (b) unchanged. REPORT.md (Summary, Finding 1, Figure 4 caption/text) and
  RESULTS.md (Figure 5 caption/text) updated to match.
- New `results/tokens_W_final_ge_large.csv`: 53 tokens (incl. ` large`) with W_final ≥ 0.4659. Summarized in
  RESULTS.md S4 (27 size words / 21 code-like strings / 4 other; block-18 count 27) and one paragraph in
  REPORT.md Finding 1.

## 2026-09-28 — feedback 3: generalization to four models (new files)
- NEW `REPORT_generalization.md` (12 figures, ~3.7k words): one section each for GPT2-XL, Pythia 1.4B,
  Llama-3.1-8B, Qwen3-8B-Base reproducing REPORT.md Figures 1–3 and the RESULTS.md S4 0.30–0.50 row,
  plus a final "does the trend generalize?" section. NEW `RESULTS_generalization.md` (G0–G5: setup, checks,
  full band and D-quintile tables, widest tokens, comparison).
- Result: distance-does-not-predict-width and single-broad-peak shape hold in all four; final block sharper
  in GPT2-XL (95.3%), Pythia (95.8%), Llama (88.8%) but not Qwen3-8B-Base (56.2%, medians 0.233 vs 0.230).
- REPORT.md, RESULTS.md and all GPT-2 Large results/plots unchanged.

## 2026-09-28 — REPORT_generalization.md: fix content-review rejection (feedback 3)
- GPT2-XL 0.30–0.50 band: "made of size and amount words" → "lowest-D examples are size and amount words", with a note that lowest-D examples do not describe the whole band.
- Pythia band: "ordinary tokens" → case/spacing variants of "big" plus unrelated pieces.
- Conclusion: removed the causal claim that peak position explains the example column; now says the column mainly reflects each model's lowest-D tokens and the cause was not tested.
- Methods precision paragraph + Qwen3 section: Qwen3's 0.003 median depth gap and 56.2% sharper share stated as near bfloat16 rounding noise. No numbers changed.

## 2026-09-28 — REPORT_generalization.md: second rejection repair (feedback 3)
- Summary 'Shape' bullet: replaced "0.30–0.50 band is the tail in the GPT-2 models but typical in the others" with per-model token shares (Pythia 66% typical, Qwen3 23% upper side, Llama 11% tail, GPT2-XL ~2% tail); Conclusion sentence aligned to the same numbers.
- GPT2-XL section: flag rate now final-block (11.5% vs GPT-2 Large 5.3%) instead of any-block 11.6% vs 5.5%, matching the Conclusion table.

## 2026-10-13 — Block-0 resid_post rerun replaces the embedding run in REPORT.md / RESULTS.md
- REPORT.md and RESULTS.md rewritten for the rerun (interpolation and distance at the last-position output of
  transformer block 0, five models). Previous versions preserved unchanged in `archive/embedding_run/`
  (REPORT.md, RESULTS.md, PLAN.md). REPORT_generalization.md / RESULTS_generalization.md are untouched and still
  describe the embedding run. New data in `results/block0_resid_post/`, plots in `plots/block0_resid_post/`;
  embedding-run data and plots not overwritten.
- Story change (old -> new, forced by Table 1/2 of the new REPORT.md): "distance does not visibly predict width at
  either block" -> "middle-block width rises with block-0 distance in GPT-2 Large and GPT2-XL, weakly in the other
  three; final block shows no shared direction".
- GPT-2 Large numbers, embedding -> block 0: median W_mid by D fifth 0.184/0.196/0.199/0.196/0.199 ->
  0.196/0.201/0.211/0.228/0.276; median W_final by fifth 0.104/0.112/0.106/0.107/0.130 ->
  0.110/0.119/0.123/0.127/0.128; median W_mid 0.195 -> 0.216; median W_final 0.112 -> 0.121; final narrower
  97.7% -> 98.4%; flagged 2,788 (5.5%) -> 1,043 (2.1%); W_final(' large') 0.466 -> 0.533; tokens at least as wide
  as ' large' 52 -> 28 others; histogram peak 0.087-0.090 -> 0.111-0.114; the 64-token spike at W_final 0.33 is gone.
- Other models, W_mid nearest -> farthest fifth (block 0): GPT2-XL 0.170 -> 0.225, Pythia 0.550 -> 0.579,
  Llama 0.339 -> 0.375, Qwen3 0.312 -> 0.335. W_final: 0.070 -> 0.075, 0.466 -> 0.427, 0.230 -> 0.266, 0.290 -> 0.312.
- Figure changes: REPORT.md now has 8 figures (GPT-2 Large Figures 1-4 regenerated; one distance-vs-width figure
  per other model). GPT-2 Large zoom window is the 5th-95th percentile of D (was the fixed 1.6-3.3 embedding range);
  representative-curve words reselected by the same rule in the new distance space.

## 2026-10-13 — Feedback 4: x range of the Llama and Qwen3 block-0 scatter plots
- `plots/block0_resid_post/gen/llama-3.1-8b_fig1_distance_vs_width.png`: x-axis full range (0.83-16.53) -> 1-2.5.
  `plots/block0_resid_post/gen/qwen3-8b-base_fig1_distance_vs_width.png`: x-axis full range (7.41-103.03) -> 10-55.
  Replotted from the existing shards; no data or measured number changed.
- REPORT.md Figures 7-8 and RESULTS.md Figures 12 and 15: captions now give the x range and the tokens not drawn
  (Llama 26 of 128,255; Qwen3 20 of 151,668). REPORT.md sentence "A few far outliers squeeze Llama's cloud into a
  narrow band" -> "Llama's cloud is a compact blob with at most a slight upward lean at the middle block".

## 2026-10-13 — Path control added (PLAN_addendum.md)
- REPORT.md: new Summary bullet, Methods paragraph "Path control (GPT-2 Large pilot)" (Matthew's path, L_path,
  y_rel equations), new Finding 5 with Table 3, one Conclusion bullet; Limitations bullet on the straight-line path
  rewritten to say the curved path and relative normalization were tried only in the pilot. Figure count unchanged (8).
- RESULTS.md: new section R6 (Table 1, Figures 18-22) and one sentence added to the Headline.
- No earlier number changed. New data `results/block0_resid_post/path_control/`, plots
  `plots/block0_resid_post/path_control/`; linear and embedding results not overwritten.
- Story change: the Conclusion's "little at the final block" is now qualified as specific to the straight path and
  dir27's normalization (final-block rise 0.02 -> 0.05 under Matthew's path, 0.10-0.14 under the relative normalization).

## 2026-10-13 — Feedback 6: summary figure of distance vs final-block width (no report change)
- New `experiments/summary_distance_vs_width.py` and `plots/block0_resid_post/summary_distance_vs_width.png` / `.pdf`.
- REPORT*.md and RESULTS*.md unchanged; no result file changed; no number superseded.

## 2026-10-13 — Feedback 6 repair: summary figure regenerated (no report change)
- `experiments/summary_distance_vs_width.py` and `plots/block0_resid_post/summary_distance_vs_width.png` / `.pdf` rewritten
  after the content review: exact [-1, 1] summary axis, full-range Llama panel with an inset, density-coloured points,
  final-block-only monotonic test stated under the figure.
- Figure counts changed with the filter (combined flag -> final-block test): GPT-2 Large 49,213 -> 49,244; GPT-2 XL
  45,953 -> 45,954; Pythia 50,155 -> 50,155; Llama 122,425 -> 122,528; Qwen3-8B 132,609 -> 132,653. Displayed r and rho
  are unchanged at two decimals.
- REPORT*.md and RESULTS*.md unchanged; no result file changed.
