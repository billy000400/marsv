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
