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

## 2026-07-17 (iter) — Fresh char + BPE Figure-9 gate verdicts (S4/S5), joint timeline + bounded verdict (S7)

- **Fresh char (30k) Figure-9 gate = FAIL.** Full 14-checkpoint LC/PGD curve (0→30000) via the
  resume-merged `results/fig9_grok_char.json` + `fig9_verdict.py`. clean acc peak 0.568@4994 → 0.554
  final; `ε=0.03` PGD adv acc → **0.528** (delayed robustness clearly emerged); test LC monotone
  1940 → **8.1** with the minimum at the *last* checkpoint → **no second descent** → FAIL.
- **Fresh BPE (10k) Figure-9 gate = FAIL.** Re-ran BPE fig9 (the finalize chain's run had OOM'd at
  `vram_frac 0.06`; re-ran at `vram_frac 0.2, pgd_bs 16, lc_bs 4`). 10 checkpoints 0→10000: clean acc
  peak 0.299@831 → 0.274; adv acc → 0.187; test LC 2182 → **95** monotone (min at last ckpt) → no
  second descent → FAIL. `results/fig9_grok_bpe.json`, `results/fig9_grok_bpe_verdict.json`.
- **Bounded relationship verdict = PLAN case 5 ("primary relationship not testable").** All three
  models (pilot char, fresh char, fresh BPE) FAIL the gate; the BPE bridge to Matthew's exact tokens
  does not reproduce Figure 9, so plateaus cannot be tied to a grokking second-descent/robustness
  window. Plateau result (char reconstruction, 14/40 pairs) is now framed as standalone, not joined
  to grokking.
- **New figures:** `plots/grokking_fresh_char.png`, `plots/grokking_fresh_bpe.png`,
  `plots/joint_timeline.png` (new `experiments/plot_joint_timeline.py`). Embedded as rendered images in
  BOTH RESULTS.md and REPORT.md.
- **Deliverables curated:** replaced the "fresh runs in progress" training-dynamics paragraph/figure
  with the completed 3-model gate table + curve figures + joint timeline + case-5 verdict in both
  RESULTS.md and REPORT.md; updated the top verdict paragraphs and REPORT Summary/Conclusion/Limitations.
- Render checks: REPORT.md 6/6 `js-display-math`, 0 `<pre lang="math">`, 0 inline-math hazards; all 8
  embedded plot paths are `![...]` images and exist on disk.
- Note: fresh char training reached step 30000 (all checkpoints saved); its post-training metadata/plot
  save crashed on an int64-not-JSON-serializable error — harmless, checkpoints intact.
- No STOP: S6 (per-checkpoint Matthew sweep / char controls) + S8 headline rework remain; case 5 makes
  S6 non-decisive but PLAN still lists it.

## 2026-07-17 (S6+S8 complete; plan finished, STOP)

- **S6 — Matthew-faithful char-token controls across training.** Ran `experiments/run_matthew_ckpts.py
  --tok char` (Matthew's exact code path: context "The house was", 50-step slerp grid, full interp-layer
  sweep, single-position patch) with the frozen controls `b↔i`,`b↔l` at the 6 frozen phases
  (`results/frozen_phases_char.json` = steps 0,56,831,7819,17500,30000). New result: **plateau emerges
  during the first LC descent** — block-0 final-logit width `w_10→90` goes 0.80 (init) → 0.35 (step 831)
  → 0.33 (step 30k), fully formed *before* ε=0.03 robustness saturates. Depth control holds
  (step 30000 b↔i: 0.33@L0 → 0.80@L11). Raw `results/matthew_char_ctrl_{raw.npz,summary.json}`; new
  figures `plots/matthew_char_ctrl_by_checkpoint.png`, `plots/joint_timeline_char_ctrl.png`.
- **S8 — de-emphasised the 40-pair reconstruction dataset.** RESULTS.md + REPORT.md rewritten so the
  Matthew-faithful `b↔i`/`b↔l` per-checkpoint assay is the **primary** plateau evidence and the 40-pair
  natural-minimal-pair sweep is a clearly-labelled **exploratory** corroboration (per PLAN out-of-scope).
  40-pair figures relabelled "Figure 3/4/5 (exploratory)". Headline/Summary/Conclusion/Limitation-4
  updated: bounded verdict unchanged (**PLAN case 5, primary relationship not testable**) but now refined
  with the S6 secondary observation — **no visible temporal coupling** between the plateau (emerges with
  initial fit) and the grokking signature (never occurs).
- **Deliverable framing:** both files retitled "Do Grokking and Matthew-style activation plateaus emerge
  together?". Render checks pass: REPORT.md 6/6 display-math, 0 broken, 0 un-rendered plot paths, 0
  inline-math hazards, all figures on disk; RESULTS.md same (no display math by design).
- **Plan complete:** all stages S1–S8 done; every success criterion met. `STOP` written (no unaddressed
  feedback files remain).

## 2026-07-25 (operator feedback #3 — comma vs every other character)

- **Addressed `human_feedback_3.txt`** ("interpolate from comma to all other characters and see if
  there is a plateau; add a section to discuss the results; do not invent jargon") → renamed
  `human_feedback_3.txt.addressed.md`.
- **New experiment** `experiments/comma_sweep.py` + `experiments/plot_comma_sweep.py`: endpoint A
  fixed at `"The house was ,"`, endpoint B = the same context + each of the **64 other characters**;
  otherwise identical to the Matthew-faithful char-control path (50 interpolation steps,
  `slerp_rescale`, final-position patch, final-logit `d(t)`, transition width `w_10→90`, frozen
  plateau rule). Run at interpolation block 0 for the 6 frozen checkpoints and at every interpolation
  block 0–11 at step 30,000. Raw curves `results/comma_sweep_raw.npz`; widths + endpoint statistics
  `results/comma_sweep_summary.json`.
- **New numbers (nothing superseded — this is a new experiment).** Final checkpoint, block 0, final
  logits: median width **0.340** (IQR 0.305–0.409), min 0.245 (`c`), max 0.665 (`3`), straight-line
  reference 0.80; **1/64** pairs meet the strict plateau rule (`w ≤ 0.25` + rests near both
  endpoints), 33/64 at ≤0.35, 52/64 at ≤0.45, **0/64** near-linear, 64/64 monotone. Median transition
  start/end t = 0.252 / 0.603. Class medians: lower-case 0.313 (n=26), upper-case 0.355 (n=26),
  space/newline 0.336 (n=2), punctuation-or-digit 0.564 (n=10). Width vs the model's next-character
  probability: Spearman **ρ = −0.74** (p = 2.7e-12, n = 64); vs endpoint logit separation ρ = −0.48
  (p = 5.6e-5). Depth: 0.34 (block 0) → 0.51 → 0.65 → 0.72 → 0.77 → 0.79 → ≈0.80 (blocks 6–11).
  Training: 0.799 (init) → 0.751 (56) → 0.524 (831) → 0.328 (7,819) → 0.367 (17,500) → 0.340 (30k).
  Context check: the preregistered `b↔i` (0.331) / `b↔l` (0.330) controls sit at this sweep's median.
- **New figures (embedded as rendered images in BOTH RESULTS.md and REPORT.md):**
  `plots/comma_all_chars_curves.png`, `plots/comma_width_by_char.png`,
  `plots/comma_width_vs_endpoints.png`, `plots/comma_depth_and_training.png`.
- **Deliverables curated:** RESULTS.md gains a "Comma vs every other character — 64 pairs from one
  endpoint" section with a 6-point discussion, plus a 4th verdict bullet and an extended headline;
  REPORT.md gains a Methods subsection (sweep design, next-character probability, endpoint separation,
  Spearman ρ — each with a rendered equation) and a Results section with the four figures and the
  discussion, plus updated Summary, Conclusion and Limitation 2. **Bounded relationship verdict is
  unchanged (PLAN case 5)** — the sweep runs on the same non-grokking checkpoints.
- Render checks: REPORT.md 9/9 `js-display-math`, 0 `<pre lang="math">`, 0 KaTeX errors, 0 inline-math
  hazards; all embedded plot paths are `![...]` images and exist on disk (both files).

## 2026-07-25 (2) — CVD-safe figures (CLAUDE.md rule 13) + context control for the comma sweep

- **All 14 deliverable figures regenerated to be readable with red-green colour deficiency.** New
  `experiments/cvd_style.py` (green-free palette `#0072B2/#D55E00/#CC79A7/#56B4E9/#E69F00`, shared
  reference-line styles). Patched `plot_fig9.py`, `plot_matthew_ckpts.py`, `plot_joint_timeline.py`,
  `plot_comma_sweep.py`, `run_matthew.py` plot section, and added `plot_training_curves.py` (redraws
  `plots/training_curves.png` from `results/train_hist.json` without retraining). Fixed violations:
  local-complexity train/test/**random** used matplotlib C0/C1/**C2 green** against a **red** dashed
  adversarial-accuracy line; the comma-sweep character classes used **green** (upper-case) vs **red**
  (punctuation); the layerwise figure used a **red** final-logit line over a viridis ramp. Now every
  series carries a linestyle, marker or hatch in addition to colour, and the sequential ramps are
  viridis/cividis. **No numbers changed** — `run_matthew.py` was re-run end-to-end and reproduced
  bit-exactly (14/40 plateaus, median w 0.309, identical depth medians).
- **Captions in RESULTS.md and REPORT.md rewritten** so no series is identified by colour ("train
  (blue), test (orange), random (green)" → "train (solid), test (dashed), random (dash-dot)", etc.),
  and a new REPORT.md Methods subsection **"Figure conventions"** states the encoding rules.
- **Corrected two stale labels:** the joint-timeline legend said "Fresh BPE (30k)" and REPORT.md's
  Summary/Conclusion said "fresh 30k-step BPE run"; the BPE run was stopped at **10k** steps (as the
  gate table already stated). Now consistent everywhere.
- **New experiment — context control (`experiments/context_sweep.py`, `plot_context_sweep.py`).**
  Repeats the operator-requested comma→every-other-character sweep in **8 further 64-character
  contexts** drawn from held-out validation text, chosen to span the model's probability of a comma
  in that slot (5e-20 … 0.997); 9 contexts × 64 pairs = **576 pairs** at step 30,000, interpolation
  block 0, final logits, all other settings unchanged. Implementation checks pass (prefix_err 0.0,
  endpoint err 1.3e-5, d(0) ≤ 1.3e-6, d(1) ≥ 0.999999). Raw `results/context_sweep_raw.npz`,
  summary `results/context_sweep_summary.json`.
- **New numbers.** 0/576 curves near-linear (w ≥ 0.70); per-context median width 0.313–0.436 (pooled
  0.381); 11/576 meet the strict w ≤ 0.25 rule, 198/576 at w ≤ 0.35. Within-context Spearman ρ
  (width vs the model's probability of the target character) is negative in **9/9** contexts (sign
  test p = 0.004; individually p < 0.05 in 7/9) with median **−0.41**, range −0.05 … −0.74; pooled
  over all 576 pairs ρ = −0.23. Median width vs p(comma) across contexts: ρ = −0.32, p = 0.41 (n = 9).
- **Claim refined (old → new).** The comma sweep's "sharpness tracks the model's next-character
  probability, **ρ = −0.74** (n = 64)" is now reported as a **range**: −0.74 is the strongest of nine
  contexts; the typical context gives **−0.41** and the pooled value is −0.23. The direction is what
  replicates. The shape claim is *strengthened* instead: "no pair is linear" goes from 0/64 in one
  context to **0/576 across nine**. The standing caveat that the fixed comma endpoint is implausible
  and might make every pair harder is **retired** — the context where a comma is 99.7% likely gives
  median width 0.330 vs the reference context's 0.340, and p(comma) does not predict sharpness.
- **New figures embedded in BOTH RESULTS.md and REPORT.md:** `plots/context_widths.png`,
  `plots/context_rho.png`. RESULTS.md gains a "Does the plateau depend on the context?" section;
  REPORT.md gains a Methods subsection (context selection + the sign-test equation) and Results
  Figures 10–11, plus an updated Summary, Conclusion and a new Limitation 5.
- **Grokking verdict unchanged: PLAN case 5** (all three Figure-9 gates still FAIL; the context
  control runs on the same non-grokking checkpoint).
- Render checks: REPORT.md 10/10 `js-display-math`, 0 `<pre lang="math">`, 0 KaTeX errors, 0
  inline-math hazards; 16 embedded images in each deliverable, all present on disk.

## 2026-07-26 (operator feedback #4 — REPORT.md math did not render on GitHub)

- **Fixed the broken equation in REPORT.md Methods.** The next-character-probability definition used
  `\operatorname{softmax}`, which **GitHub's math renderer refuses** — it replaces the whole equation
  with the red error *"The following macros are not allowed: operatorname"*, so that definition was
  invisible to any reader on GitHub. Now `\mathrm{softmax}`, which renders. Same paragraph: inline
  `$x_{ctx}$` → `$x_{\text{ctx}}$` so the context subscript is upright rather than reading as a
  product of three variables. No numbers, figures or claims changed.
- **This was a regression of operator feedback #1** (same `\operatorname` complaint, addressed
  2026-07-17). It reappeared because the feedback-#3 iteration wrote a new Methods paragraph, and the
  render check in use at the time (GitHub-API `js-display-math` placement + an inline-hazard grep)
  cannot catch it: the LaTeX is *valid*, so placement passes and KaTeX itself compiles it — only
  GitHub's macro denylist rejects it. Hence the mechanical guard below rather than another prose fix.
- **New `experiments/check_render.py` + `experiments/katex_compile.js`** — one command that fails
  loudly on all four known GitHub-rendering traps: (1) KaTeX-compiles every ` ```math ` fence;
  (2) KaTeX-compiles every inline `$…$` **after applying GitHub's backslash-before-punctuation
  stripping**, so CLAUDE.md rule-8b breaks surface as real KaTeX errors instead of silently;
  (3) flags denylisted macros (`\operatorname`, `\def`/`\newcommand`-family, `\href`/`\html*`);
  (4) confirms via the GitHub markdown API that each display equation became `js-display-math` and
  none became `<pre lang="math">`, and that no `(plots/x.png)` path is missing its `![…]` embed.
  Self-tested against a file containing one of each failure mode (5/5 caught, exit 1).
- **Both deliverables now pass** it: REPORT.md 10 display equations, 91 inline expressions, 16
  embedded figures, **0 problems**; RESULTS.md 16 embedded figures, 0 problems.
- **`../CLAUDE.md` updated** as the feedback requested, surgically: new **8c** (GitHub rejects
  `\operatorname` and the definition/HTML macro families outright — use `\mathrm`/`\text`; built-in
  operators need no wrapper) and **8d** (run one script that checks 8a–8c and rule 12; eyeballing has
  failed every time), with the one-time `npm install --prefix /tmp/katexcheck katex` setup.
- **Deliverables record the check** for the reader: a "Rendering check" paragraph in REPORT.md
  §Methods/Figure conventions and a line in RESULTS.md §Implementation checks.
- Verdicts, gate table, plateau numbers and all 16 figures per deliverable unchanged (case 5 + the S6
  secondary temporal observation). `human_feedback_4.txt` → `human_feedback_4.addressed.md`.

## 2026-08-01 — S9 / Experiment 5: all-pairs character sweep (2,080 pairs) + rule-12 caption fix

**New experiment (operator request, reopening the direction).** Every character interpolated to every
other character, and a stated hypothesis about what the plateaus correspond to.

- **New code.** `experiments/allpairs_sweep.py` (all C(65,2)=2,080 pairs at interpolation block 0 of
  the step-30,000 fresh-char checkpoint; also the step-0 init re-run, a 200-pair depth subsample at
  blocks 4/8/11, and a 100-pair endpoint-swap replication), `experiments/analyze_allpairs.py`
  (per-character statistics, variance decomposition, readout-decision test, correlations),
  `experiments/plot_allpairs.py` (six figures). `experiments/matthew_assay.py` gained two minimal
  additions used by the new test — the per-`t` `argmax` trace in `run_pair`, and `iso_crossing`
  factored out of `transition_width` so `t*` uses the identical crossing rule. No assay, score,
  interpolation scheme or step grid was changed; `self_test()` still passes.
- **New artifacts.** `results/allpairs_raw.npz` (per-pair `d(t)` + per-`t` argmax, final and init),
  `results/allpairs_summary.json` (per-pair, per-character, analysis block),
  `plots/allpairs_{width_matrix,width_by_char,curves_small_multiples,boundary_vs_logp,readout_decision,controls}.png`.
- **New results in both deliverables (Figures 14–19).** All 2,080 pairs pass the endpoint checks
  (max `d(0)`=3e-6, min `d(1)`=0.999998, prefix error exactly 0.0) and every curve is exactly monotone.
  Swap symmetry: median and max |Δw| = **0.000** over 100 pairs — reported as an algebraic identity
  (`d(t)`→`1−d(1−t)` on a symmetric grid) that the check confirms, which licenses the symmetric heatmap.
  Median width **0.355** (IQR 0.298–0.444); strict rule 182/2,080 (8.8%); near-linear 20/2,080 (1.0%).
- **Per-character verdict = PLAN case (i)**: basin fraction ≥ 0.86 for **all 65** characters (1.00 for
  59), per-character median widths 0.264 (`o`) – 0.590 (`3`), and the additive fit `w_ij ~ a_i + a_j`
  explains **78.2%** of the width variance (adjusted 77.6%) against a 3.0% permutation chance level —
  so cases (ii) and (iii) are ruled out: sharpness travels with the character, not the pair.
- **Mechanism.** Readout-decision test: **91%** of next-character prediction changes fall inside the
  transition window, **79%** of pairs have all changes inside it, **80%** have single-prediction flat
  arms, median |t*−t_flip| = 0.045 (2.2 grid steps); paths visit a median of 3 predictions (32% exactly
  2). Controls: at init **all** 2,080 paths are straight (median width 0.803 → 0.355 trained, 0 strict,
  Mann–Whitney p < 1e-300), and the sharpness is generated by blocks 1–4 (median width 0.344 at block
  0, 0.763 at 4, 0.806 at 8 and 11 = the straight-line value). Plausibility confound retested at
  n = 2,080: ρ(w, max p) = −0.46 and ρ(w, endpoint separation) = −0.46, both surviving partial
  correlation at −0.59; per-character ρ(med_w, log p) = −0.60.
- **New "What do the plateaus correspond to?" subsection** in both deliverables with the required
  4-sentence hypothesis: a plateau is the set of final-position residual states that decode to the same
  next-character prediction, one basin per character, built by blocks 1–4; the named live alternative
  is that plausibility carves the basin; the falsifiable follow-up is rebalancing the unembedding to
  equalise p(A) and p(B) (decision account: t* → 0.5, width unchanged).
- **Grokking side unchanged.** All three Figure-9 gate verdicts, the case-5 relationship verdict and
  the S6 timing observation stand exactly as before; PLAN forbids this series revising them, and no
  number in those sections moved.

**Rule-12 fix applied to both deliverables (standing operator complaint, named in `../CLAUDE.md`).**
Previously all 16 captions per file lived in the `![...]` **alt text**, which GitHub only shows when an
image fails to load — so both files rendered as unlabelled images. Every figure now has short alt text
plus a **visible `**Figure N.**` caption line below the image**, and figures are **renumbered
sequentially in reading order** (previously REPORT.md ran 1, 1b, 1d, 2a, 2b, 6–11, then 3–5 at the end,
with two embeds unnumbered). Both files: 22 embeds, 22 visible captions, numbering 1…22, each figure
cited by number in the prose and preceded by a sentence naming the claim it evidences.
`python3 experiments/check_render.py REPORT.md RESULTS.md` → **0 problems** (REPORT.md 17 display
equations, 200 inline expressions, 22 figures). Methods gained definitions with rendered equations for
every new metric (median width, basin fraction φ, strict fraction σ, the additive variance
decomposition R², t*, t_flip, partial Spearman) and a new baseline entry for the initialization sweep
and the permutation chance level. No previously reported number was changed by the renumbering.
