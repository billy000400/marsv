# CHANGELOG — Direction: Do the 12-layer Shakespeare GPT's activations show plateaus?

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-15 — Full study, first (and final) result: calibrated NEGATIVE

Went from empty scaffold to a complete go/no-go verdict in one iteration.

- **S1 source audit.** Audited the official repo `AhmedImtiazPrio/grok-adversarial` via GitHub API:
  it releases only MNIST-MLP and CIFAR-ResNet training code — **no GPT/Shakespeare code or
  checkpoint**. So the Figure-9 GPT is tested as a faithful reconstruction (success-criterion 3
  qualifier). Wrote `MODEL_SPEC.md` (every field tagged confirmed/reconstruction).
- **S2 model.** Trained reconstruction 12L/12H GeLU char GPT (d_model=240, ctx 128, 8.38M params) on
  Tiny Shakespeare → **val loss 1.494, next-char acc 0.560** (≈37× chance). `plots/training_curves.png`.
- **S3 assay.** Implemented final-position residual intervention + PI/sharpness/JSD; unit test detects
  a synthetic plateau (PI +0.33) and scores a line (PI 0.00); alpha=0 reconstruction check passes
  (<1e-3 logit err).
- **S4/S5 pilot + confirmatory (all blocks 0–10).** 48 held-out contexts × 8 directions × 41 radii,
  natural vs matched-control basepoints. **Result: median PI(natural) NEGATIVE at every block
  (−0.15 to −0.30) → saturating, not plateaued.** ΔPI(nat−ctrl) small but positive & significant at
  all blocks (peak +0.096, Cliff's δ +0.91 at blocks 2–3); JSD agrees in sign. flip@max ≥0.81.
- **RESULTS.md / REPORT.md** written from scratch to current-best. Figures embedded:
  training_curves, response_by_layer, plateau_score_by_layer, individual_curves. All 6 display-math
  blocks verified rendering via the GitHub markdown API; no inline-math hazards; no unrendered paths.
- **Verdict:** NO plateaus detected (qualified reconstruction) → **no-go** for a plateau-mapping
  follow-up on this model. Direction complete; `STOP` written.

## 2026-07-16 — Curation pass for updated operator rules; STOP re-created

No unaddressed feedback files in this direction (checked: none matching `human_feedback*`/`*REVIEW*`).
Re-entered because the previous iteration's `STOP` was never persisted to disk (no trace in git) and
the operator relaunched with updated CLAUDE.md rules. No experiments re-run; results unchanged.

- **REPORT.md Methods rewritten to the new rule-9 narrative form:** every metric now has a
  motivation ("what question does this answer, why not the obvious alternative") *before* its
  definition, and names the Result/figure that consumes it. Figures numbered (Figure 1–4).
- **Boundary sharpness was defined but consumed by no Result (new rule: such a metric gets cut) —
  fixed by reporting it instead:** added a sharpness paragraph to REPORT.md Results and a
  `sharp nat / ctrl` column to the RESULTS.md table (from `results/confirm_summary.json`, values
  nat 2.16–4.01 vs ctrl 2.99–4.91; linear reference 1.0, synthetic plateau 3.2). Interpretation:
  with PI < 0 the steep segment is the initial rise, not a late plateau edge, and natural rays are
  *less* sharp than control everywhere — corroborates the negative verdict.
- Re-verified per CLAUDE.md 8a/8b/12: 6/6 display equations render via the GitHub markdown API,
  0 `<pre lang="math">`, no inline-math hazards, no unrendered plot paths.
- Verdict unchanged: NO plateaus (qualified reconstruction), no-go. `STOP` written (this time verified
  on disk).
