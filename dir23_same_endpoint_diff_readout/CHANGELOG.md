# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-08-12 — first and final content for REPORT.md and RESULTS.md

Both files went from template stubs to complete deliverables; no earlier numbers were superseded.

- **REPORT.md** — new. Research question, methods (prompt, five readout suffixes, slerp
  interpolation, endpoint JSD, normalized logit distance d(t), t10/t50/t90, width w, Δt50), seven
  embedded figures (immediate prediction; the five individual d(t) curves; transition comparison),
  Table 1, and the verdict. Headline: Δt50 = 0.011 across Capital (t50 0.454), Continent (0.444),
  Currency (0.443) and Language (0.450) → **aligned transitions**. Type control discussed separately.
- **RESULTS.md** — new. S1 endpoint reproduction against the PLAN's preliminary table (all values
  match: immediate JSD 0.00761 bits; readout JSDs 0.991 / 0.885 / 0.915 / 0.968 / 0.111 bits), full
  transition table with crossing counts and monotonicity, top-1 flip positions (t = 0.44–0.47), the
  immediate-position d(t) caveat, the overlay figure, and the data/code index.
- **plots/** — new: distance_{capital,continent,currency,language,type}.png, distance_overlay.png,
  immediate_prediction.png, transition_comparison.png.
- **results/** — new: s1_endpoints.json, interp.csv, interp.npz, transitions.json.

---

## 2026-09-23 — Figure 7 gains the top-1 token along the sweep (human_feedback_0.txt)

- **plots/transition_comparison.png (REPORT.md Figure 7)** — redrawn as two panels (full sweep and a
  0.40–0.50 zoom). Each readout keeps its t50 marker and [t10, t90] bar and now also has a row of 101
  squares giving the top-1 token at each t (open = Japan-side answer, filled = Germany-side answer),
  plus a dashed line at the observed top-1 switch.
- **REPORT.md** — Methods: new metric t_switch (zero crossing of p(a_A) − p(a_B)) with the term
  "delayed logits" defined. Results: section retitled, Figure 7 caption rewritten, new Table 2
  (switch vs t50). Summary and Conclusion updated. Old summary quoted first-new-token grid points
  (0.46 / 0.44 / 0.47 / 0.45) as flip positions; replaced by interpolated t_switch 0.460 / 0.432 /
  0.461 / 0.447, with offsets from t50 of +0.006 / −0.011 / +0.018 / −0.003 (Currency lags).
  Word count ~3.8k, still 7 figures.
- **RESULTS.md** — top-1 table extended with bracketing grid points, t_switch, offset and d(t_switch);
  data index lists the new `results/top1_tokens.csv`.
- **experiments/s3_plots.py** — transition-comparison block rewritten; writes `results/top1_tokens.csv`.
  No model re-run: top-1 ids and answer probabilities were already saved in `results/interp.npz`.
