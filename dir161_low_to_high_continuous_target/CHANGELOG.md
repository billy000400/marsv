# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-28 — first and complete deliverables (S1–S4)

Direction created from the PLAN.md design; RESULTS.md and REPORT.md went from placeholder stubs to
full current-best deliverables. Nothing was superseded (no prior numbers existed).

**What was run.** S1 protocol audit (`experiments/audit.py`), S2 matched training for seeds 0/1/2
(`train.py`) plus task-adequacy and baseline evaluation (`evaluate.py`), S3 the frozen 90-pair SLERP
probe at both checkpoint conditions (`probe.py`), S4 aggregation, bootstrap and figures
(`analyze.py`), plus a committed per-pair metric export (`export_pairs.py`).

**New results in RESULTS.md / REPORT.md.**
- Removed-detail audit: `r = P(y)` carries 39.1% of pixel energy; operator identities `D(U(z))=z`,
  `D(P(y))=0`, `UD+P=I` hold to <=2.4e-07.
- Classifier gate: top-1 accuracy 95.8 / 96.3 / 96.8% (seeds 0/1/2) on the untouched pool — passes.
- Predictor gate: full-image MSE 0.0137 and removed-detail MSE 0.0136 vs block repeat 0.0401/0.0401
  and bicubic 0.0346/0.0323; `R2_detail` 0.660 [0.654, 0.666] vs bicubic 0.195 and the privileged
  digit template 0.165 — passes, and beats the privileged template by 0.0198 [0.0195, 0.0202]
  detail MSE, so the recovered detail is instance-specific.
- Primary plateau comparison (linearity deviation, best-val checkpoints, classifier − predictor):
  hidden 2 0.0968 [0.0873, 0.1063] (5.4x), hidden 3 0.1335 [0.1197, 0.1474] (4.9x), output
  0.1403 [0.1300, 0.1510] (4.1x); predictor smoother on 90/90 pairs at every layer, every seed
  positive. Max normalized jump agrees (hidden 3: 2.86 vs 1.24, diff 1.62 [1.44, 1.80]).
- Robustness: fraction normalization and final step-30,000 checkpoints both keep every interval
  above zero; endpoint reproduction error <=1.4e-06; all probe reruns bit-identical.
- **Verdict recorded: robust positive** under the preregistered rule.

**Figures added and embedded (10, in reading order, in BOTH deliverables):** `data_audit.png`,
`training_curves.png`, `superres_panel.png`, `baseline_bars.png`, `hand_selected_curves.png`,
`mean_curves.png`, `paired_difference.png`, `per_pair_scatter.png`, `path_predictions.png`,
`checkpoint_control.png`. `experiments/check_render.py REPORT.md RESULTS.md` exits 0 (13 display
equations, 84 inline, 10 embeds, 0 problems per file).
