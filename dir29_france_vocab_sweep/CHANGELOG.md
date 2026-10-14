# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-09-29 — first full deliverables
- REPORT.md written from template: histogram of W_final over all 50,256 tokens (Figure 1), 40 narrowest tokens
  (Table 1; full W ≤ 0.3 list in `results/france_W_le_0.3.csv`), d(t) curves on both sides of 0.3 (Figure 2),
  30 widest tokens with manual labels (Table 2), conclusion.
- RESULTS.md written: counts, token-list paths, inspection counts, reference-token percentiles, curve crossings.
- Framing: the PLAN labels W ≤ 0.3 as "non-plateau", but Figure 2 shows that small W is the flat-then-jump plateau
  shape (Direction 27's definition). The report gives the literal W ≤ 0.3 list and also names the W > 0.3 side as the
  smooth transitions. No superseded numbers.

## 2026-10-14 — Figure 1 restyled (human_feedback.txt, human_feedback_2.txt)
- `plots/france_vocab_width_histogram.png` redrawn in the style of panel (a) of Direction 27's
  `fig2_final_width_distribution.png`: 6 x 4.5 in (was 6.4 x 3.8), 200 bins (was 100), linear y-axis (was log), no dashed
  W = 0.3 line and no count annotations, title `"The capital of [France]" -> [B] sweep`.
- REPORT.md and RESULTS.md: Figure 1 caption updated to match (linear scale, 200 bins, no dashed line). No numbers changed;
  the counts on each side of W = 0.3 stay in the prose and tables.
- Code: the figure is now drawn by the new `experiments/plot_sweep.py` from `results/france_tokens.csv`; the histogram
  block was removed from `experiments/analyze.py`. `PLOTTING.md` documents script, data and command.

## 2026-10-14 — review repair for human_feedback_2.txt
- REPORT.md: the sentence introducing Figure 1 no longer says the plot shows the two sides of the threshold (the restyled
  figure draws no threshold line); it now says the counts on each side of W = 0.3 are in the text below. No numbers changed.
- PLOTTING.md: explains what Direction 27 is and writes out the colour rules instead of pointing to `../CLAUDE.md`.
- RESULTS.md, the PNG and `experiments/plot_sweep.py` unchanged.

## 2026-10-14 — second review repair for human_feedback_2.txt (text only)

- RESULTS.md 'Headline': "smooth switches are mostly place names" -> "among the 100 widest switches, 53 are place
  names and 17 are scraped tokens; the other 3,083 tokens with W > 0.3 were not labelled". No number changed.
- REPORT.md: Direction 27, PLAN.md and `resid_post` are defined at first use. The 93.7% / 94% share with W ≤ 0.3 now
  cites `results/summary.json`; Figure 1 is cited only for the shape of the distribution.
- RESULTS.md header says what Direction 27 is. PLOTTING.md 'Data' says what a shard is and where the shards are.
- No figure, table, analysis or result file changed.
