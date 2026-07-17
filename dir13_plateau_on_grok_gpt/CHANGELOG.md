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

## 2026-07-17 — Operator feedback #1 addressed: banned macro removed, REPORT.md tightened

Feedback file `human_feedback_1.txt` (now `.addressed.md`) asked: (1) the `\operatorname` macro is
not allowed in REPORT.md; (2) per the updated CLAUDE.md, make REPORT.md more readable and concise.
No experiments re-run; all numbers and the verdict unchanged.

- **ΔPI equation fixed:** `\operatorname{median}(…)` → `\mathrm{median}(…)` (and `\text{…}`
  subscripts → `\mathrm{…}`) in the Group-comparison display equation. Zero `operatorname` hits
  remain in REPORT.md/RESULTS.md.
- **REPORT.md rewritten more concise/readable** (215 → 197 lines, shorter sentences, redundancy cut)
  while keeping every CLAUDE.md rule-8/9 requirement: narrative Methods with per-metric motivation
  and consuming figure, all 6 display equations, baselines, calibration list, jargon defined on
  first use (grokking, residual stream, JSD, Cliff's delta), and axis/legend descriptions added to
  every figure caption (Figs 1–4).
- Re-verified: 6/6 display equations render via the GitHub markdown API, 0 `<pre lang="math">`,
  no inline-math hazards, no unrendered plot paths. RESULTS.md untouched (no banned macros; already
  concise — feedback's "it" read as REPORT.md).
- Verdict unchanged: NO plateaus (qualified reconstruction), no-go. `STOP` re-created after feedback
  was addressed.

## 2026-07-17 — REOPENED plan executed: Matthew-style two-endpoint assay REVERSES the verdict

PLAN.md was rewritten by the operator to reopen the direction: the previous random-ray result
(2026-07-15, "no plateaus") answered a different question — one-sided perturbation response along
random directions — and is NOT evidence about Matthew-style two-natural-endpoint plateaus. That old
assay (PI/sharpness/JSD, `plots/response_by_layer.png`, `plots/plateau_score_by_layer.png`,
`plots/individual_curves.png`, `results/confirm_*.json`, `results/tidy_results.csv`) is now history
recorded here only; it was removed from RESULTS.md/REPORT.md per PLAN S7.

New experiment (S3–S6, all run this iteration):
- **S3:** froze 40 natural minimal pairs (`experiments/make_pairs.py`, seed 20260717) from held-out
  val text: prefix len 127 + endpoint char; A = observed next char, B = model top-1 (top-2 if ==A);
  0 excluded by the frozen degeneracy threshold (endpoint logit dist < 1e-3); dists 8.7–64.4.
  Saved `results/prompt_pairs.json` before inspecting any curve.
- **S4:** `experiments/matthew_assay.py` — norm-interpolating slerp (clamped cos, documented
  near-collinear fallback), final-position patching via exact partial forward, Matthew relative
  distance d(t), PAVA-isotonic transition width w_10→90, frozen plateau rule (w≤0.25, t_lo≥0.10,
  t_hi≤0.90, iso-dev≤0.10). Self-tests: synthetic step w=0.089 detected, line w=0.800 rejected;
  d(0)=0, d(1)=1; slerp endpoint/norm identities.
- **S5/S6:** `experiments/run_matthew.py` — checks passed on real pairs (endpoint logit err <1e-3,
  prefix activation match <1e-4, batched=single-example <1e-5), then primary (block 0 → logits,
  101 t), layerwise (resid_post blocks 1–11), depth comparison (interp blocks {0,2,4,6,8,10}).
  Saved `results/matthew_tidy.csv`, `results/matthew_summary.json`.

**Verdict SUPERSEDED: "NO plateaus (random-ray)" → "YES, Matthew-style plateaus present (qualified
reconstruction)".** Numbers: 14/40 pairs pass the frozen rule in raw individual final-logit curves
(0 non-monotone; median w 0.309, range [0.110, 0.773]; 24/40 at w≤0.35; 2 near-diagonal); layerwise
median w falls monotonically 0.777 (block 1) → 0.445 (block 11) → 0.309 (logits), strict rule passes
only at logits; depth comparison rises 0.309 → 0.802 ≈ diagonal 0.8 at block 10 — both predicted
plateau signatures. Go for the plateau-mapping follow-up.

- RESULTS.md and REPORT.md rewritten from scratch around the new assay; new figures embedded:
  `pair_curves_logits.png` (all 40 raw curves), `layerwise_emergence.png` (pairs 0–3, frozen choice),
  `interpolation_layer_comparison.png`; `training_curves.png` retained for provenance.
- Render checks: 4/4 display equations js-display-math via GitHub markdown API, 0 pre-lang-math,
  no inline-math hazards, no unembedded plot paths.
- PLAN.md S3–S7 ticked, status COMPLETE (plateaus present); `STOP` written after verifying zero
  unaddressed feedback files.

## 2026-07-17 — Operator feedback #2 addressed: "two signatures" sentence clarified; grok-phenomenon scope stated

Feedback file `human_feedback_2.txt` (now `.addressed.md`) raised two points about the Summary:
(1) the sentence "Two independent signatures behave exactly as predicted … sharpens through layers …
fades toward the diagonal …" was confusing — it never said where "the diagonal" lives or what it is;
(2) the Summary never said whether the trained model replicates the phenomenon introduced in the Grok
paper. No experiments re-run; all numbers and the verdict unchanged.

- **REPORT.md Summary rewritten:** the diagonal is now defined where it is first used — the straight
  line `d = t` that the relative-distance curve traces when the output morphs uniformly (transition
  width 0.8, no plateau segments) — and both signatures are phrased as movements of the *same* curve
  relative to that diagonal: deeper recording moves the curve away from it; later interpolation
  collapses the curve back onto it. The curve's axes (`d`, `t`) are glossed inline so the Summary is
  self-contained before Methods.
- **Grok-phenomenon scope now explicit in the Summary:** the grok paper's own phenomenon (grokking =
  adversarial robustness emerging long after training accuracy saturates) is NOT tested or replicated
  — our reconstruction trains only to ordinary convergence; the paper's role is solely to specify the
  Figure-9 model. This was previously implicit in Limitation 3 only.
- **RESULTS.md verdict paragraph updated in parallel:** same diagonal definition + signature phrasing,
  and the same explicit grokking-not-tested note.
- Re-verified per CLAUDE.md 8a/8b/12: 4/4 REPORT.md display equations render as js-display-math via
  the GitHub markdown API, 0 `<pre lang="math">`, no inline-math hazards, no unembedded plot paths.
- Verdict unchanged: Matthew-style plateaus present (qualified reconstruction), go. `STOP` re-created
  after the feedback was addressed.

## 2026-07-17 — PLAN re-dropped in REOPENED state; full assay re-run, bit-exact reproduction; deliverables unchanged

Re-entered to find `STOP` cleared and PLAN.md replaced (uncommitted) with the same reopen-plan text
already executed earlier today ("(2)" entry above) — S3–S7 unchecked, status claiming no slerp
experiment had run, contradicted by `results/matthew_summary.json`/`matthew_tidy.csv` on disk. The
dropped file was also truncated (missing the plan's title/research-question/setup sections) and
contained no requirement differing from the executed plan, so it was treated as a stale re-drop
(assumption logged in JOURNAL) and answered by **re-verification rather than blind re-ticking**:

- Re-ran the complete pipeline (`experiments/run_matthew.py`): corpus SHA-256 re-verified against
  training provenance, self-tests (step w=0.089 detected / line w=0.800 rejected), implementation
  checks on real pairs (endpoint <1e-3, prefix <1e-4, batch=single <1e-5), then primary + layerwise +
  depth runs on the 40 frozen pairs.
- **Reproduction is bit-exact:** summary JSON identical to the frozen one — 14/40 plateau pairs, same
  IDs (0,4,5,6,7,9,14,20,21,22,28,34,36,37), median w 0.309 [0.110, 0.773], 0 non-monotone, depth
  medians 0.309→0.802 identical to all digits. No tracked artifact changed.
- RESULTS.md / REPORT.md: **no content changes** (already current-best; numbers verified, not
  superseded). Render checks re-pass: 4/4 REPORT.md display equations as js-display-math via the
  GitHub markdown API, 0 `<pre lang="math">`, no inline hazards, no unembedded plot paths.
- PLAN.md status/checkboxes restored to COMPLETE (edited in place on the operator's dropped file;
  its truncated preamble left as dropped).
- Verdict unchanged: Matthew-style plateaus present (qualified reconstruction), go. `STOP` re-created
  after verifying zero unaddressed feedback files.

## 2026-07-17 (5) — S3 Figure-9 validity gate added to deliverables (pilot = FAIL)

Context: PLAN was reopened to the full Grokking-replication scope (S3–S7). Prior iterations set up the
source-locked Fig. 9 pipeline and started fresh char+BPE trainings; this iteration lands the first
Grokking-side result into the deliverables.

- **New result — pilot Figure-9 gate.** Ran `experiments/fig9.py` on all 13 log-spaced checkpoints of
  the existing 3,500-step char pilot (`results/fig9_pilot_char.json`), then the preregistered
  `experiments/fig9_verdict.py`. Verdict = **FAIL** within the 3,500-step horizon: clean acc → 0.564
  (peak at last ckpt), `ε=0.03` PGD adv acc → 0.327 (delayed robustness *did* emerge), but test LC
  falls monotonically 1940 → 68 with its minimum **at the final checkpoint** — **no second LC descent**.
  Saved `results/fig9_pilot_char_verdict.json`; plot `plots/grokking_pilot_char.png`.
- **RESULTS.md:** added a "Figure-9 grokking gate (validity gate — S3 pilot)" section (metric prose,
  verdict table, embedded plot); rewrote the top-verdict scope note from "grokking not tested here" →
  "grokking now measured as a gate; pilot FAILs within horizon; fresh runs training; plateau result not
  yet joined to a Grokking claim."
- **REPORT.md:** Summary scope note updated identically; new Methods §"Figure-9 grokking gate" defining
  **local complexity** and **`ε=0.03` PGD adversarial accuracy** with rendered equations plus the
  preregistered PASS/NOT-ESTABLISHED/FAIL rule; new Results paragraph + embedded Figure 1b.
- Render checks: **6/6** REPORT.md display equations render as `js-display-math` via the GitHub markdown
  API, 0 `<pre lang="math">`, 0 inline-math hazards (fixed one `\,`→`\thinspace`), 0 unembedded plots.
- Plateau numbers unchanged (still current-best). No STOP (plan incomplete: S4/S5/S6/S7 pending; fresh
  char/BPE trainings + their fig9 evals running in background).

## 2026-07-17 (5) — Fresh-run training-dynamics figure; queued late-checkpoint char gate eval

- **New figure `plots/fresh_training_dynamics.png`** (`experiments/plot_fresh_training.py`, parses the
  live training logs): fresh char + BPE runs' train/val loss and val next-token accuracy vs step. Both
  runs **overfit** — val loss bottoms early (char 1.47 @ step ~3,750; BPE 4.77 @ step ~750) then rises
  while train loss keeps falling; val acc plateaus (char ≈0.56, BPE ≈0.27). Opposite of grokking's
  delayed val-loss recovery, foreshadowing (not yet deciding) a likely gate FAIL.
- **RESULTS.md + REPORT.md:** added an "in progress" fresh-run training-dynamics paragraph and embedded
  the new figure (RESULTS.md; REPORT.md as Figure 1c). Clearly marked that the decisive per-checkpoint
  LC/PGD gate eval is not yet complete.
- **Late-checkpoint char fig9 eval confirmed queued.** A prior iteration already launched
  `/tmp/chain_char_late.sh` (waits for char training + early eval, then resume-merges late steps incl.
  the final checkpoint into `results/fig9_grok_char.json`) so a *second* LC descent can be detected. I
  killed a duplicate chain I had briefly started (would have been a second concurrent writer to the
  shared JSON) and kept the pre-existing one.
- Render checks: REPORT.md 6/6 `js-display-math`, 0 `<pre lang="math">`, 0 inline-math hazards, 6
  embedded plot images; RESULTS.md 6 embedded images, 0 bare-path refs, 0 hazards.
- Plateau numbers and pilot gate verdict unchanged (still current-best). No STOP (S4/S5/S6/S7 pending;
  fresh trainings + fig9 evals running in background).
