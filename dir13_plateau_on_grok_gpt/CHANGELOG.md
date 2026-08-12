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

## 2026-08-02 — Corrected Figure-9 verdict detector (operator feedback `human_feedback_4.txt`): two runs flip FAIL → PASS

**Feedback addressed** (`human_feedback_4.txt` → `human_feedback_4.txt.addressed.md`). The operator
reported that the character runs do show the Figure-9 pattern and that `fig9_verdict.py` was buggy.
Confirmed: the old rule located the LC minimum with `np.argmin(lc)`, i.e. the **global** minimum, which
in every one of our runs is the **last checkpoint** (end of the second descent) — so no rise could ever
be found after it and every run was scored "no second descent".

**Code.** `experiments/fig9_verdict.py` rewritten to detect the structure in order: (1) first
*significant* local minimum (earliest interior dip whose following rise clears tol = 5% of the LC
range); (2) the subsequent local maximum, which defines the second descent's **onset**; (3) a
*sustained* second descent after it (falls > tol, no rebound to the horizon, ends below the first
minimum). Added the two preregistered ordering checks that were missing: onset precedes the
clean-accuracy peak, and adversarial robustness rises by ≥ 0.05 from its value at the onset (plus a
*sustained* robustness onset — first checkpoint from which adv acc stays ≥ 0.05 — so a one-checkpoint
transient cannot count). The old `final_adv_acc >= 0.05` test is gone. Rerun on the **existing** JSONs
only; no training was extended.

**Superseded numbers (old → new).**
| run | verdict | second descent | new landmarks |
|---|---|---|---|
| Pilot char (3.5k) | **FAIL → PASS** | No → **Yes** | LC 1940 → 484 @ 19 → 1043 @ 33 → 68; onset 33 < acc peak 3500; adv 0.000 → 0.327; sustained adv onset 1091 |
| Fresh char (30k) | **FAIL → PASS** | No → **Yes** | LC 1940 → 491 @ 15 → 769 @ 56 → 8.1 (rise 278 vs tol 96.6, CI ±3); onset 56 < acc peak 4994; adv 0.0006 → 0.530; sustained adv onset 831 |
| Fresh BPE (10k) | FAIL (unchanged) | No | only upturn 459.5 @ 56 → 489.2 @ 217 = 30 units = 1.4% of range, below the 5% tolerance |

**Deliverables.** RESULTS.md and REPORT.md rewritten wherever the gate was reported: Summary/headline,
Methods "Preregistered verdict rule" (now states the three-step detector and both ordering checks as
equations), the gate section and its table (old row "second LC descent? No/No/No" → the landmark rows
above), Figure 2–4 captions, the joint-timeline paragraph and Figure 5 caption, Conclusion and
Limitation 4. **Bounded relationship verdict:** primary (Matthew-exact `big/in`, `big/large`, BPE only)
stays **PLAN case 5, not testable**, because the BPE run is the one that still FAILs; the character
analogues move from "no visible temporal coupling" to **PLAN case 1, temporally associated** — the
`b↔i`/`b↔l` plateau sharpens from width 0.80 to 0.33 between steps 56 and 831, inside the second-descent
window (56 → 30,000) and across the sustained robustness onset (831). Stated with its limits: one run,
six checkpoints, and an onset early enough (step 56) that the window overlaps ordinary initial fitting.

**Figures.** `plots/grokking_pilot_char.png`, `plots/grokking_fresh_char.png`,
`plots/grokking_fresh_bpe.png` regenerated with the detected landmarks annotated (grey rules + ▽/△
markers + the verdict), and `plots/joint_timeline.png` regenerated (its verdict panel now reads
PASS / PASS / FAIL). Figure numbering and the 22 embeds in each deliverable are unchanged.

**Render checks.** `node` is unavailable this session so `check_render.py` could not run; the individual
checks were run instead — GitHub markdown API: 20 `js-display-math`, 0 `<pre lang="math">` (equals the
20 ` ```math ` fences); rule-8b inline hazard grep: clean (one new inline `\_` was moved into a fence);
22 embeds / 22 visible `**Figure N.**` captions in each file; no bare `(plots/*.png)` paths.

## 2026-08-02 (later) — densified the fresh-char Figure-9 grid: the LC local maximum is not a one-point artifact

**Why.** The corrected gate (earlier today) upgraded the fresh character run to PASS on the strength of
an LC dip-then-rise whose local maximum was defined by **one** log-spaced checkpoint (step 56). Both
deliverables carried that as an explicit caveat. It was also PLAN's own named next follow-up ("a denser
checkpoint grid over steps 10–1000 to resolve the LC local maximum with more than one point"). This
iteration ran it.

**What ran.** `experiments/fig9.py` on 10 checkpoints the fresh char run had **already saved** but never
evaluated — steps 1, 2, 6, 9, 23, 36, 88, 138, 339, 531 — identical pipeline, identical frozen
evaluation points (seed 20260717), identical `r=0.005`, `P=25`, 1024 points, `ε=0.03` PGD. No training
extended, no threshold changed, detector untouched. Fresh char grid **14 → 24 checkpoints**. Two
incidental fixes: `/tmp/tinyshakespeare.txt` had been cleared, so the corpus was re-fetched and its
SHA-256 verified equal to `train_meta_grok_char.json`'s `corpus_sha256`
(`86c4e6aa…dc565ed`); and the script's default `--vram_frac 0.05` OOMed, so it was run at the
BUDGET-allowed 0.225 with the PGD batch size unchanged (changing the batch would have perturbed the
attack's random starts and broken comparability with the existing 14 points).

**Superseded numbers (old → new), fresh char (30k) run only. Verdict unchanged: PASS.**
| quantity | old (14 ckpts) | new (24 ckpts) |
|---|---|---|
| points above the first LC minimum | 1 | **3** (steps 23, 36, 56) |
| LC local maximum | 769.4 ± 3.0 @ step 56 | **989.1 ± 4.5 @ step 36** |
| LC rise above first minimum | 278 units (2.9× tol) | **498 units (5.1× tol)** |
| second-descent onset | step 56 | **step 36** |
| adv acc at onset | 0.0006 | 0.0012 |
| sustained robustness onset (adv ≥ 0.05 thereafter) | step 831 | **step 531** (step 531 = 0.077, previously unmeasured) |
| 5%-of-range tolerance | 96.59 | 96.83 |

The new intermediate LC values are 1945 @ 1, 1899 @ 2, 1123 @ 6, 747 @ 9, 988 @ 23, 989 @ 36, 604 @ 88,
494 @ 138, 329 @ 339, 192 @ 531. The first local minimum (491.2 @ 15) and the final value (8.1) are
unchanged, as are the pilot-char and BPE verdicts (PASS / FAIL) and their grids.

**Deliverables.** RESULTS.md and REPORT.md updated everywhere the fresh-char gate is reported: summary
bullets, gate table (new row "points resolving the LC local maximum"; checkpoints 14 → 24), the gate
paragraph, Figure 3's caption, the joint-timeline paragraph, Conclusion and Limitation 4. REPORT.md
Methods §Figure-9 gate now states the checkpoint grid per run and that densification changed only the
grid. The caveat "the local maximum is resolved by a single log-spaced checkpoint in each run" is now
scoped to the **pilot** run only, which is still true of it. The relationship verdicts are unchanged —
primary **PLAN case 5** (BPE still FAILs), character analogues **PLAN case 1** — but the association
window is restated as second descent 36 → 30,000 with sustained robustness onset 531, so the `b↔i`/`b↔l`
sharpening interval (56 → 831) now *strictly contains* the robustness onset instead of ending at it.

**Figures.** `plots/grokking_fresh_char.png` regenerated on the 24-point grid (the dip-and-rise is now a
resolved V-then-Λ notch rather than a single-point spike) and `plots/joint_timeline.png` regenerated so
its LC/adv panels use the denser curve. No new figure files; embeds and numbering unchanged (22 each).

**Raw.** `results/fig9_grok_char.json` (24 records), `results/fig9_grok_char_verdict.json`,
log `results/fig9_grok_char_dense.log`; pre-densification copies kept as `*.bak`.

## 2026-08-02 (later still) — readout-rebalancing intervention: the plateau is upstream of the decision

**What changed.** New experiment (PLAN's named next step, the one intervention that could separate the
two live accounts of the character basins): `experiments/rebalance_probe.py` +
`experiments/plot_rebalance.py`. It adds a constant to one row of the unembedding output — a pure
readout bias that leaves every residual-stream activation on the interpolation path bit-identical —
and asks whether the plateau boundary follows the decision boundary it moves. Run on all 1,873 of the
2,080 character pairs whose two endpoints predict different next characters (207 predict the same
character at both ends and have no boundary), at interpolation block 0 of the step-30000 fresh-char
checkpoint, same frozen assay (`matthew_assay.run_pair`, 50-step slerp, final-logit `d(t)`). The only
code change to the frozen path is an opt-in `return_logits=False` argument to `run_pair`, off by
default, so no existing result is affected.

**New numbers.** `d(t)` is exactly invariant to an additive readout bias (measured deviation
1.3e-6 = float32 noise), so `w_10→90` and `t*` cannot be moved by the readout at all. The readout gap
swings a median 21.9 nats along the path, so the boundary is stiff: an equalising bias of 2.44 nats
moves it a median 0.020 in `t` (80% of pairs < 0.05), and the 5.28-nat bias that forces the boundary to
the path midpoint moves it a median 0.052. Median `|t* − t_gap|` = 0.025 unmodified → 0.015 equalised →
0.035 midpoint-forced.

**Superseded interpretation (no superseded numbers).** The S9 hypothesis previously ended with a
*promise* ("a follow-up could separate the two…") and cited the median `|t* − t_flip|` = 0.045
alignment as support for the decision account. That alignment is now stated as correlational, not
causal: both the prediction flip and the `d(t)` transition are driven by the same sharp residual-stream
change built by blocks 1–4, and the readout is a steep but passive reader of it. The hypothesis
sentence survives as a *description* of the basins, not as their mechanism. The intervention also
turns out to be structurally unable to test the plausibility account's width prediction — no
readout-level bias can change `d(t)` — which is now stated as a limit of the test, with the conclusion
that plausibility, if it acts, must act through the learned weights of blocks 1–11.

**Deliverables.** RESULTS.md: new subsection "The readout-rebalancing intervention — the plateau is
upstream of the decision" replacing the follow-up promise. REPORT.md: matching Results subsection plus
a new Methods block defining the readout gap `g(t)`, the decision boundary `t_gap`, and the two frozen
biases `c_eq`, `c_half` with ```math fences at column 0. New **Figure 20** embedded in both with a
visible caption; the three exploratory figures renumbered 20–22 → 21–23 so numbering stays sequential
in reading order (23 embeds, 23 captions in each file).

**Raw.** `results/rebalance_summary.json` (per-pair rows + summary), `results/rebalance_raw.npz`,
`plots/rebalance_readout.png`.

## 2026-08-02 (later) — MLP-gain intervention: blocks 1–4 causally set the plateau sharpness

**What ran.** `experiments/mlp_gain_probe.py` + `experiments/plot_mlp_gain.py`. The successor the
readout probe named: instead of biasing the readout (which cannot move `d(t)` at all), scale the
MLP-branch output of a block group by a gain `g` and re-run the frozen assay with endpoints recomputed
under the modified model. 150 random pairs (seed 0) of the 2,080, interpolation block 0, step-30000
fresh character checkpoint — same setting as Experiment 5, so widths are directly comparable. Groups:
blocks 1–4 (early, implicated by the depth control) and blocks 8–11 (late, specificity control);
gains 0 / 0.5 / 1.5 plus the unmodified model. No training, no threshold, no assay change.

**New numbers.** Early group, median `w_10→90`: **0.796** (g=0) → 0.533 (g=0.5) → 0.351 (unmodified)
→ **0.305** (g=1.5); strict plateau rate 0.000 / 0.000 / 0.100 / **0.300**. At g=0 the width is back at
the untrained value 0.803 and **every** pair widens (frac Δw>0 = 1.00, median Δw = +0.433). Late group:
median `w` 0.337 / 0.333 / 0.380 for g = 0 / 0.5 / 1.5, median paired |Δw| ≤ 0.025 (17× smaller at g=0
than the early group). Median |Δt*| 0.074 at g=0, ≤ 0.024 elsewhere — sharpness moves, location does not.

**Superseded.** Nothing numerically. Experiment 5's depth result (w 0.34/0.76/0.81/0.81 at patch blocks
0/4/8/11) stated where the sharpness is *observed* to be built; that is now an intervention, so the
report says blocks 1–4 *causally* set the sharpness. The plausibility alternative is narrowed (it must
act through those same early weights) but explicitly not eliminated.

**Deliverables.** New subsection "The MLP-gain intervention — blocks 1–4 causally set the sharpness" in
RESULTS.md and REPORT.md (the latter also gets a Methods block defining the gained-block update with a
column-0 ```math fence), new **Figure 21** embedded in both with a visible caption, and one added
sentence each in the REPORT Summary and Conclusion. The three exploratory figures renumbered 21–23 →
22–24 so numbering stays sequential in reading order (24 embeds, 24 `**Figure N.**` captions per file).

**Raw.** `results/mlp_gain_summary.json`, `results/mlp_gain_raw.npz`, `plots/mlp_gain_intervention.png`.

## 2026-08-02 (latest) — per-block MLP scan: the sharpness is distributed, and tracks neither mediator

**What ran.** `experiments/mlp_block_scan.py` + `experiments/plot_mlp_block_scan.py`, the successor the
MLP-gain intervention named in PLAN: delete each of blocks 1–4's MLP branch on its own (`g=0`) and ask
(a) which block carries the sharpness and (b) whether the resulting width change tracks the endpoint
**plausibility** confound or the **decision** structure. Same frozen assay, same fixed 150-pair random
subsample (seed 0), interpolation block 0, step-30000 fresh character checkpoint — so every width is
directly comparable to the gain experiment and to Experiment 5. All-four deletion re-run in the same
script as an in-run reference. No training, no threshold, no assay change; 14 s on GPU at
`vram_frac 0.225`, 2 threads.

**New numbers.** Median `w_10→90`: 0.351 (unmodified) → 0.541 / 0.478 / 0.446 / 0.402 deleting block
1 / 2 / 3 / 4 alone → 0.796 deleting all four. As a share of the all-four effect that is **41% / 28% /
18% / 11%**, monotone in depth and summing to 98%. Strict plateau rate 10% → 0–3% singly, 0% for all
four; fraction of pairs widened 0.99 / 0.96 / 1.00 / 0.95.
Plausibility mediator (`max_p = max(p(A|ctx), p(B|ctx))`, partialled against endpoint logit
separation, both recomputed under each ablated model): partial ρ = **−0.634** unmodified — reproducing
Experiment 5's −0.587 on this 150-pair subsample — and −0.610 / −0.565 / −0.561 / −0.641 / −0.450
under the five ablations, i.e. the association survives; but it does **not** mediate the intervention:
ρ(Δw, Δmax_p) = +0.106 / +0.148 / −0.007 / +0.016 singly and **+0.221** for all four, with median
|Δmax_p| ≤ 0.0007 against a width shift of +0.433. Median `max_p` actually *rises* 0.0034 → 0.0136
under the all-four ablation, the direction that predicts *narrower* plateaus.
Decision mediator: **80.7%** of pairs still predict different characters at their two endpoints after
all four MLPs are deleted (86.7% unmodified), median distinct `argmax` regions **3** in every
condition, while median `|t* − t_flip|` decouples **0.043 → 0.214**.

**Superseded interpretation (no superseded numbers).** The Experiment-5 hypothesis previously named
endpoint plausibility as "the leading alternative these data do not rule out" and ended with a pointer
to a future test. Plausibility is now **excluded as the mechanism** (it does not mediate the ablation
and moves against the predicted direction) while **retained as a predictor** of which pairs are sharp
(partial ρ = −0.59 survives every ablation). The "decodes to the same prediction" clause is
correspondingly demoted from mechanism to description, since the decision survives the ablation that
destroys the plateau. Both deliverables' hypothesis paragraphs were rewritten to say this and now end
with the falsifiable prediction PLAN 5.5 requires (freeze blocks 1–4 at step-0 weights, train the rest
to the same validation accuracy, expect paths to stay straight). REPORT Limitation 6 was rewritten
from "the rebalancing experiment is the test that would separate them" (stale — that test has since
run) to the current state.

**Deliverables.** New subsection "The per-block scan — the sharpness is distributed, and tracks neither
plausibility nor the decision" in RESULTS.md and REPORT.md; a new REPORT Methods block defining the
per-block share `F_l`, the plausibility mediator and its mediation correlation, and the three decision
descriptors, with column-0 ```math fences. New **Figure 22** embedded in both with a visible caption;
the three exploratory figures renumbered 22–24 → 23–25 so numbering stays sequential in reading order
(25 embeds, 25 `**Figure N.**` captions per file). REPORT Summary and Conclusion each gained a
sentence.

**Render check.** `node` is present on this pod for the first time in several iterations, so
`python3 experiments/check_render.py REPORT.md RESULTS.md` ran in full for the first time since it was
written: **ALL CHECKS PASS** (REPORT 26 display eqs / 321 inline eqs / 25 figures, RESULTS 25 figures,
0 problems). It caught nothing pre-existing, but the rule-8b inline grep caught two `\,` thin spaces
in newly written inline math (`$\rho_{w,\max p\,\cdot\,\mathrm{sep}}$` and a `),\,p(B`), which GitHub
would have rendered as stray commas; both fixed.

**Raw.** `results/mlp_block_scan_summary.json`, `results/mlp_block_scan_raw.npz`,
`plots/mlp_block_scan.png`.

## 2026-08-02 (latest) — frozen-block training test: the hypothesis's own prediction is FALSIFIED

**What ran.** The training run PLAN S10 specifies and the previous iteration set up but could not
finish: `experiments/train_frozen.py` retrained the reference character recipe twice from scratch with
a block group held at its step-0 weights — `frozen_early` (blocks 1–4, the group the MLP-gain probe and
per-block scan implicate) and `frozen_late` (blocks 8–11, the specificity control) — everything else
identical (same corpus SHA, model/data seeds, Adam schedule, 30,000 steps, batch, checkpoint grid).
Both reached step 30,000 (46 min each, `vram_frac 0.11`, 1 thread). `experiments/frozen_assay.py` then
ran the frozen assay on each at its matched-accuracy checkpoint AND its final checkpoint against three
reference conditions (step 0 / 2500 / 30000) on the same fixed 150-pair subsample, plus the
injection-depth control (blocks 0/4/8) on the three final models. 36 s at `vram_frac 0.225`.

**New numbers.** Final validation next-character accuracy **0.5625** (frozen-early) / **0.5622**
(frozen-late) vs the reference run's **0.5502**; matched accuracy reached at step 2750 / 2500.
Median `w_10→90`: **0.471** (frozen-early final, IQR 0.403–0.524) and **0.484** (frozen-late final,
IQR 0.427–0.551), against 0.803 untrained, 0.443 at the reference's matched step and 0.351 at the
trained reference. Paired median `Δw` vs the trained reference **+0.107** (94% of pairs widen) and
**+0.120** (96%); vs the reference at step 2500 only **+0.033** / **+0.038**. Strict plateau rate
10% (reference) → **0.7%** / **0%**. Depth control (injection blocks 0/4/8): reference
0.351/0.761/0.805, frozen-late 0.484/0.793/0.806 (same profile), frozen-early
0.471/**0.471**/0.788 — zero width change across the frozen group, so the sharpening now happens in
blocks 5–7. Unchanged: median `t*` 0.491/0.495 (vs 0.488), endpoints differ 84%/93% (vs 86.7%),
median `argmax` regions 3, median `|t*−t_flip|` 0.062/0.059 (vs 0.043; 0.214 under the MLP ablation),
partial ρ(`w`, `max_p` | sep) −0.61/−0.60 (vs −0.634).

**Superseded interpretation (no superseded numbers).** Both deliverables' hypothesis paragraphs ended
on the prediction "freeze blocks 1–4 at step-0 weights, train the rest to matched accuracy, expect the
paths to stay straight (≈0.80)". That prediction is **falsified**: 0.471 recovers 73% of the reference
run's sharpening, and the specificity control costs just as much width (0.484), so the shortfall is a
generic capacity cost of freezing four blocks rather than anything about blocks 1–4. The claim "blocks
1–4 build the sharpness" is therefore narrowed from a training-time necessity to an inference-time fact
about this trained network; the sharp transition is a **relocatable** computation. Both hypothesis
paragraphs now say this and end with a new falsifiable prediction (freeze blocks 1–7, train only the
top of the stack: sharpening should reappear between injection blocks 8 and 11). REPORT Limitation 6
updated accordingly and a new Limitation 7 added for the frozen test's own scope (one seed per
condition, two frozen groups, three injection depths).

**Deliverables.** New subsection "The frozen-block training test — the sharpness does not have to be
learned in blocks 1–4" in RESULTS.md and REPORT.md; new **Figure 23** embedded in both with a visible
caption defining all four panels' axes and series; REPORT Methods §Frozen-block training test gained a
sentence defining the depth control it uses; REPORT Summary, REPORT Conclusion, RESULTS verdict item 5
and the RESULTS Headline each gained the frozen-block result. Exploratory figures were already numbered
24–26 by the previous iteration, so numbering stays sequential (26 embeds, 26 `**Figure N.**` captions
per file). `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
(REPORT 28 display / 350 inline eqs / 26 figures; RESULTS 26 figures; 0 problems).

**Raw.** `results/frozen_assay_summary.json`, `results/frozen_assay_raw.npz`,
`results/frozen_assay.log`, `results/train_hist_frozen_{early,late}.json`,
`plots/frozen_blocks.png`. Checkpoints live in gitignored scratch at `/tmp/dir13_frozen/`.

## 2026-08-02 (S11) — deep-freeze training test: the relocation prediction is CONFIRMED

**What ran.** `experiments/train_frozen.py --freeze 1,2,3,4,5,6,7 --tag frozen_deep --steps 30000`
(the run pre-launched at the end of the previous iteration) finished the full 30,000 steps, and
`experiments/frozen_assay.py` was re-run with `frozen_deep` in its condition list and the
injection-depth grid extended from blocks 0/4/8 to **0/4/8/10/11**. Same frozen assay, same fixed
150-pair subsample, same context, same seed — only the condition list and the depth grid changed.
`experiments/plot_frozen.py` re-rendered `plots/frozen_blocks.png` with the third run added.

**New result (Figure 23, both deliverables).** Prediction on record: with blocks 1–7 held at
initialization the paths should still sharpen well below the untrained 0.80, with the width drop
appearing between injection blocks 8 and 11. **Confirmed.** Final validation next-character accuracy
**0.5742** — the highest of any run, reference 0.5502 — matched accuracy at step 3000. Median width
**0.558** (IQR 0.471–0.621), narrower than untrained for 149/150 pairs (Wilcoxon p = 2e-26), i.e. 54%
of the reference sharpening recovered with 58% of the blocks never updated. Depth profile
0.558 / 0.557 / 0.695 / 0.767 / 0.805 at injection blocks 0/4/8/10/11: the frozen blocks 1–4 contribute
−0.002 and the entire 0.248 of sharpening sits in the four trainable blocks (0.139 across blocks 5–8,
of which only block 8 can train; 0.071 across 9–10; 0.039 in block 11). Geometry otherwise unchanged
(t* 0.486, endpoints differ 87.3%, median 3 argmax regions, |t*−t_flip| 0.092, partial rho −0.62);
strict plateau rate 0%.

**New cross-run finding.** The width cost tracks the *number* of frozen blocks, not their depth:
0.351 (none) → 0.471 / 0.484 (four) → 0.558 (seven). Paired: frozen-deep wider than frozen-early by
+0.073 (83% of pairs, p = 1e-17) and than frozen-late by +0.064 (80%, p = 1e-16), while frozen-early
and frozen-late differ by only −0.015.

**Corrections to numbers already published.** (a) The injection convention is that patching at block
`b` replaces `resid_post` of block `b`, so the width drop between injection points `b1 < b2` is
produced by blocks `b1+1 … b2`. Frozen-early's relocation was therefore reported one block too narrow:
**"relocated to blocks 5–7" → "relocated to blocks 5–8"** everywhere in both deliverables (the
underlying numbers 0.471 @4 vs 0.788 @8 are unchanged). (b) The reference and frozen-early/late depth
profiles gained their blocks 10 and 11 entries: reference 0.805 @8 → **0.806 @10, 0.805 @11**;
frozen-early 0.788 @8 → **0.804 @10, 0.809 @11**; frozen-late 0.806 @8 → **0.806 @10, 0.806 @11** —
i.e. nothing above block 8 contributes in any of them, which is what makes frozen-deep's rise across
9–11 informative.

**Superseded interpretation.** The previous hypothesis paragraphs ended on the prediction that freezing
blocks 1–7 would either relocate the sharpening again or leave the paths straight. It relocated, so
both paragraphs now state the general finding — the sharp transition is a relocatable computation whose
sharpness degrades with the number of frozen blocks, not their depth — and end on a **new** falsifiable
prediction: freeze blocks 5–11 (the mirror image, same count frozen, trainable capacity at the bottom).
If count is what matters the width should land near 0.558 with the whole drop between injection blocks
0 and 4; if depth matters it should differ markedly.

**Deliverables.** RESULTS.md subsection retitled "The frozen-block training test — the sharpening
relocates into whatever blocks stay trainable" and rewritten around a three-run prediction/outcome
table; the matching REPORT.md Results subsection rewritten the same way; Figure 23's caption updated
for six top-row panels and five injection depths in both files; REPORT Methods §Frozen-block training
test now defines the third run, the extended depth grid and the span-attribution rule; REPORT Summary,
REPORT Conclusion, REPORT Limitations 6–7, RESULTS Headline and RESULTS verdict item 5 all curated to
current-best. Figure count unchanged at 26 embeds / 26 `**Figure N.**` captions per file.
`python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS** (REPORT 28 display /
373 inline eqs / 26 figures; RESULTS 26 figures; 0 problems).

**Raw.** `results/frozen_assay_summary.json` and `results/frozen_assay_raw.npz` (both now carry all
nine conditions), `results/train_hist_frozen_deep.json`, `plots/frozen_blocks.png`. Checkpoints remain
in gitignored scratch at `/tmp/dir13_frozen/checkpoints_frozen_deep/`.

## 2026-08-02 (S12, same iteration as S11) — mirror-image freeze: relocation confirmed a third time, "count not depth" falsified

**What ran.** `experiments/train_frozen.py --freeze 5,6,7,8,9,10,11 --tag frozen_mirror --steps 30000`
— the mirror image of `frozen_deep`: the same 4.86M of 8.38M parameters frozen (58.0%) and the same
five trainable blocks, but at the *bottom* of the stack (blocks 0–4 trainable) instead of the top
(blocks 0, 8–11). `experiments/frozen_assay.py` gained the condition and injection block **2** (so the
drop can be resolved inside the trainable group), giving a depth grid of 0/2/4/8/10/11 for all five
depth conditions; `experiments/plot_frozen.py` was generalised to render one top-row panel per frozen
run present (7 panels now) and re-rendered `plots/frozen_blocks.png`.

**New result (Figure 23, both deliverables).** Prediction on record: if width is set by the *count* of
trainable blocks rather than their depth, frozen-mirror should land near frozen-deep's 0.558 with its
entire width drop between injection blocks 0 and 4 and nothing above. **Split verdict.** The location
half is confirmed exactly — depth profile 0.626 / 0.764 / **0.805** / 0.806 / 0.806 / 0.806 at
injection blocks 0/2/4/8/10/11, i.e. injecting at block 4 already gives the untrained straight line and
all sharpening sits in blocks 1–4 (0.138 in blocks 1–2, 0.042 in 3–4). The magnitude half is falsified:
median `w` **0.626** (IQR 0.555–0.681), not 0.558 — paired median Δw vs frozen-deep **+0.063**, 81% of
pairs wider, p = 6e-17, i.e. **39%** vs 54% of the reference sharpening recovered. Final validation
accuracy **0.5744** (matched at step 2750), the highest of any run and within 0.0002 of frozen-deep's,
so this is not a capacity or task-performance difference. Geometry otherwise unchanged (t* 0.499,
endpoints differ 86.7%, |t*−t_flip| 0.085, partial rho −0.54); strict plateau rate 0%.

**Superseded numbers/claims from the S11 entry above.** (a) "The width cost tracks the *number* of
frozen blocks, not their depth" → replaced by: **trainable depth is the first-order term, position a
second-order one that only bites once depth is scarce.** With 8 trainable blocks position is worth
0.015 (frozen-early 0.471 vs frozen-late 0.484); with 5 it is worth 0.068 (frozen-deep 0.558 vs
frozen-mirror 0.626), favouring blocks near the readout. Series across runs: 0.351 (12 trainable) →
0.471/0.484 (8) → 0.558/0.626 (5). (b) The depth grid gained block 2, so three published profiles
gained a point: reference 0.351/**0.646**/0.761/…, frozen-early 0.471/**0.471**/0.471/…, frozen-late
0.484/**0.739**/0.793/…, frozen-deep 0.558/**0.558**/0.557/… — showing the reference's sharpening is
front-loaded into blocks 1–2, matching the per-block MLP scan's 41/28/18/11% shares.

**New falsifiable prediction** replacing S11's (which this run answered): if trainable depth really is
the first-order term, freezing *ten* blocks (training only block 0 and block 11) should land straighter
still, near 0.70, with its residual drop split between injection blocks 0→2 and 10→11; if instead one
trainable block adjacent to the readout suffices, it should come out near 0.56.

**Deliverables.** Both frozen-block subsections rewritten around a four-run prediction/outcome table
with the new depth-vs-count paragraph; Figure 23 caption updated for seven top-row panels, six
injection depths and five accuracy curves in both files; REPORT Methods §Frozen-block training test now
defines the fourth run and why it isolates position from count; REPORT Summary, REPORT Conclusion,
REPORT Limitations 6–7, both hypothesis paragraphs, RESULTS Headline and RESULTS verdict item 5 all
curated to current-best. Figure count unchanged at 26 embeds / 26 `**Figure N.**` captions per file.
`python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS** (REPORT 28 display /
383 inline eqs / 26 figures; RESULTS 26 figures; 0 problems).

**Raw.** `results/frozen_assay_summary.json` and `results/frozen_assay_raw.npz` (now eleven
conditions), `results/train_hist_frozen_mirror.json`, `plots/frozen_blocks.png`. Checkpoints in
gitignored scratch at `/tmp/dir13_frozen/checkpoints_frozen_mirror/`.

---

## 2026-08-02 (S13) — Two-block freeze: the trainable-depth prediction confirmed, and the first run in which the plateau breaks

No unaddressed `human_feedback*`/`*REVIEW*` files (all five end in `.addressed.md`), so this iteration
advanced the plan: the S12 successor prediction, tested.

**New experiment.** `experiments/train_frozen.py --freeze 1,2,3,4,5,6,7,8,9,10 --tag frozen_two` —
82.9% of the parameters (6.94M of 8.38M) held at their step-0 weights, leaving only blocks 0 and 11
trainable, everything else identical to the reference character run. 19 minutes of training, then
`frozen_assay.py` (which already carried the condition entry) on the same fixed 150 pairs.

**Prediction on record → outcome.** Predicted ≈0.70 if trainable depth is the first-order term, ≈0.56
if one trainable block beside the readout suffices. Outcome **w = 0.726** (IQR 0.642–0.802) →
**trainable-depth prediction CONFIRMED**, the one-block alternative excluded. Final val acc **0.5668**
(above the reference's 0.5502); matched at step **7000** vs 2500–3000 for every other frozen run.
Paired shifts: **+0.160** vs frozen-deep (97% of pairs, p = 7e-26), **+0.094** vs frozen-mirror (89%,
p = 3e-21), **+0.363** vs the trained reference (99%, p = 2e-26).

**The half of the prediction that could not have come true.** It also said the residual drop would
split between injection blocks 0→2 and 10→11. The measured profile is
0.726/0.725/0.724/0.725/0.725/**0.803** at blocks 0/2/4/8/10/11 — the entire 0.077 sits in 10→11, i.e.
block 11 alone. Reason, now stated in both deliverables: injecting at block 0 *overwrites* block 0's
output, so block 0's trainable weights are invisible to the measurement and block 11 is the only
trainable block downstream of it. Recorded as a methods point, not a surprise about the network.

**Superseded numbers.** Nothing was re-measured, so no prior number changed; the cross-run summary
sentence was extended, old → new: "0.351 → 0.47–0.48 → 0.56–0.63 for 12, 8 and 5 trainable blocks" →
"0.351 → 0.47–0.48 → 0.56–0.63 → 0.726 for 12, 8, 5 and effectively 1 usable block". Both deliverables'
framing of the frozen series changed from "freezing only costs *how* sharp it gets" to that claim plus
an explicit floor, because frozen-two is the first condition where the geometry itself fails: 26% of
its pairs are *wider* than untrained (0–1% in the other four runs), |t*−t_flip| 0.146 vs 0.043,
partial ρ −0.18 vs −0.63, strict_frac 0, and only 17% of the reference sharpening recovered.

**New falsifiable prediction** replacing S12's (which this run answered): frozen-two confounds
trainable depth with parameter count, so the next test holds depth fixed and cuts capacity — retrain at
`n_embd` 192 instead of 384 with nothing frozen. Depth account → ≈0.35 like the reference; parameter
count account → ≈0.47 like the eight-trainable-block runs.

**Deliverables.** Both frozen-block subsections rewritten around a five-run prediction/outcome table,
with a new paragraph/bullet on the two-block run and where the plateau breaks; Figure 23 re-rendered
(eight top-row panels, six injection curves, six accuracy curves) and its caption updated in both
files; the bottom-left panel title corrected from "Freezing a block group slows the sharpening but does
not prevent it" to "…blunts the sharpening; only the 1-10 run nearly removes it", which the new
condition made the old title overclaim; REPORT Methods already defined the fifth run and needed no
change; REPORT Summary, REPORT Conclusion, REPORT Limitations 6–7, both hypothesis paragraphs, RESULTS
Headline and RESULTS verdict item 5 all curated to current-best. Figure count unchanged at 26 embeds /
26 `**Figure N.**` captions per file. `python3 experiments/check_render.py REPORT.md RESULTS.md` →
**ALL CHECKS PASS** (REPORT 28 display / 408 inline eqs / 26 figures; RESULTS 26 figures; 0 problems).

**New code.** `experiments/frozen_pairwise.py` — paired median shift + Wilcoxon signed-rank *between*
frozen runs, read off the per-pair widths already stored in `frozen_assay_raw.npz` (the assay itself
only pairs each condition against the three reference conditions). Validated by reproducing S12's
published frozen-mirror vs frozen-deep numbers exactly (+0.0633, 81.3%, p = 6.1e-17).

**Raw.** `results/frozen_assay_summary.json` and `results/frozen_assay_raw.npz` (now thirteen
conditions), `results/frozen_pairwise.json`, `results/train_hist_frozen_two.json`,
`plots/frozen_blocks.png`. Checkpoints in gitignored scratch at
`/tmp/dir13_frozen/checkpoints_frozen_two/`.

## 2026-08-02 (S14 — narrow run: trainable depth vs trainable capacity de-confounded)

- **New experiment.** `experiments/train_frozen.py` gained `--n_embd` and now accepts an empty
  `--freeze`, so the same harness trains the **narrow** control: `n_embd` 192, nothing frozen, every
  other setting identical to the reference character run. 5,584,896 parameters vs the reference's
  8,378,640 — within 0.3% of frozen-early's 5,601,360 *trainable* parameters, but with all 12 blocks
  trainable. New `experiments/narrow_assay.py` scores just this condition with the frozen assay's own
  functions and merges the row into `results/frozen_assay_summary.json` / `frozen_assay_raw.npz`;
  `experiments/frozen_pairwise.py` gained its three matched-accuracy comparisons.
- **Result (new, nothing superseded).** At matched accuracy (step 2,750, val 0.5543) the narrow run's
  median transition width is **0.397** (IQR 0.311–0.526) — the depth account's prediction (≈0.35–0.44),
  falsifying the capacity account's ≈0.47. Paired over the same 150 pairs: **−0.073** vs frozen-early
  (23% of pairs wider, p = 2.5e-15), **−0.092** vs frozen-late (13%, p = 1.8e-19), and **−0.014** vs the
  reference at its own matched step (39% wider, p = 1.9e-4), i.e. slightly sharper. Depth profile
  0.397 / 0.569 / 0.686 / 0.763 / 0.807 / 0.832 at injection blocks 0 / 2 / 4 / 8 / 10 / 11 (the
  reference's front-loaded shape); partial ρ = −0.65; strict-rule fraction **13.3%**, the only run
  besides the reference to retain the sharpest tail (frozen runs: 0–0.7%).
- **RESULTS.md / REPORT.md.** New bullet/paragraph in both frozen-block subsections; new Methods
  paragraph in REPORT defining the narrow run and both accounts' point predictions as a rendered
  equation. New **Figure 24** (`plots/capacity_vs_depth.png`) embedded with a visible caption in both
  files. The three exploratory figures were renumbered 24–26 → **25–27** in captions and prose so
  figures stay in reading order. Figure 23 is unchanged: the narrow run is not a frozen-block run and
  a seventh series would have exceeded the five-colour CVD palette.
- **Reading updated.** The five-run summary "trainable depth first, position second" now rests on a
  matched control for the confound it carried: depth is the variable, parameter count is not.

## 2026-08-02 (S14b — narrow run scored at the end of training; Figure 24 gains both framings)

- **New measurement (nothing superseded).** The narrow run (`n_embd` 192, nothing frozen) finished
  training and `experiments/narrow_assay.py` scored its final checkpoint, adding the
  `narrow192_last` row that S14 left open. Median transition width **0.332** (IQR 0.288–0.389) at step
  27,143, validation accuracy 0.5639, over the same 150 pairs at interpolation block 0. Paired
  (`experiments/frozen_pairwise.py`, four comparisons added): **−0.010** vs the reference's
  fully-trained 0.351 (43% of pairs wider, Wilcoxon p = 2.1e-4), **−0.124** vs frozen-early's 0.471
  (1.3%, p = 2.6e-26), **−0.146** vs frozen-late's 0.484 (3.3%, p = 3.6e-26), and **−0.065** against its
  own matched-accuracy row (23%, p = 3.1e-14). Depth profile 0.332 / 0.626 / 0.746 / 0.794 / 0.802 /
  0.808 at injection blocks 0 / 2 / 4 / 8 / 10 / 11; strict-rule fraction 12.0% (reference 10.0%);
  partial ρ = −0.51. Caveat recorded with the number in both deliverables: the harness time budget
  stopped this run at 27,143 of 30,000 steps (lr annealed to 1.2e-4 rather than 1.0e-4), which can only
  understate its final sharpness since it was still sharpening.
- **Figure 24 re-rendered** (`experiments/plot_capacity.py`): each run now shows its matched-accuracy
  point (large marker) *and* its end-of-training width (small open square, dotted connector, new legend
  entry), the two 12-block runs are separated along the left panel's x-axis so they no longer overlap,
  the subtitle states both framings, and the legend moved to the left panel's free corner. CVD-safe as
  before — no red/green, family carried by colour AND marker AND fill.
- **RESULTS.md / REPORT.md.** New bullet (RESULTS) / paragraph (REPORT) reporting the fully-trained row
  and its caveat; Figure 24's caption rewritten in both files for the two-framing figure; REPORT Methods
  extended to state that the narrow run is also assayed at its final checkpoint and that it stopped at
  27,143 steps.
- **Two stale claims corrected in REPORT.md.** Limitation 7 still read "separating them needs a
  narrower-but-full-depth run, **which was not performed**" — it was performed in S14; it now names the
  narrow run and states the remaining limitation (single seed, so the 0.397-vs-0.476 gap has no
  across-seed error bar). The Experiment-5 "What this settles" paragraph's closing caveat was corrected
  the same way. The REPORT **Summary** did not mention the depth-versus-capacity result at all; it now
  carries it in one sentence with both numbers (0.397 matched, 0.332 trained).
- **Raw.** `results/frozen_assay_summary.json` and `results/frozen_assay_raw.npz` (fourteen conditions),
  `results/frozen_pairwise.json`, `plots/capacity_vs_depth.png`. `narrow_assay.py` is now genuinely
  idempotent per condition key (it skipped the already-published `matched` row instead of re-scoring
  it). Figure count 27 embeds / 27 captions per file; `python3 experiments/check_render.py REPORT.md
  RESULTS.md` → **ALL CHECKS PASS** (REPORT 29 display / 437 inline eqs / 27 figures; RESULTS 27
  figures; 0 problems).

## 2026-08-02 — S14c: a second seed for the narrow run (across-seed error bar on the depth conclusion)

- **New run.** `train_frozen.py` gained a `--seed` flag (model init seed only; data order unchanged);
  a second `n_embd` 192 run with nothing frozen was trained from model seed 2024 and scored with
  `narrow_assay.py narrow192_s2` on the same 150 pairs. It reached the reference's final validation
  accuracy 0.5502 at the same step 2,750 (val 0.5547) as seed 1337.
- **RESULTS.md / REPORT.md (Experiment 5).** New result: narrow seed 2 median transition width
  **0.437** (IQR 0.326–0.514, strict rule 10.7%) against seed 1337's 0.397, i.e. an across-seed spread
  of ≈0.04 (paired Δw +0.015, p = 0.015). Both seeds stay below frozen-early (seed 2: −0.044, 33% of
  pairs wider, p = 2.7e-8) and frozen-late (−0.062, 20%, p = 1.6e-16); two-seed mean 0.417 vs the
  capacity account's ≈0.47 and the depth account's ≈0.35–0.44. The depth-over-capacity conclusion is
  unchanged, now with the spread quantified.
- **Sub-claim retracted (old → new).** "At matched accuracy the narrow run is slightly *sharper* than
  the full-width reference (−0.014, p = 1.9e-4)" → the second seed is indistinguishable from the
  reference (−0.004, 46% of pairs wider, p = 0.17), so the deliverables now say narrowing costs
  nothing measurable rather than that it helps. REPORT Summary and Results wording updated to match.
- **Figure 24 (`plots/capacity_vs_depth.png`) regenerated** with the second narrow seed as a third
  filled circle at 12 trainable blocks (and nudged apart from seed 1 on the parameter axis); label
  offsets reworked because the three all-trainable runs overplotted. Caption in both deliverables now
  reads the gap between the two filled circles at 12 blocks as the across-seed spread.
- **Limitation 7 updated (old → new).** "the narrow run … is a single seed, so the 0.397-versus-0.476
  gap has no error bar across seeds" → the spread is now bounded at ≈0.04, and the remaining
  single-seed caveat is narrowed to the five frozen conditions.
- Caveat recorded in the deliverables: this second run was stopped after its matched-accuracy
  checkpoint (wall clock), so it contributes a matched-accuracy row only, no end-of-training row.

## 2026-08-02 — Finalization: deliverables verified, direction closed

- **No content changes to RESULTS.md or REPORT.md.** Both were already curated to current-best in the
  S14c iteration; this pass verified them rather than rewriting them, since a 127 KB prose rewrite
  under a 20-minute finalization budget risks regressions of exactly the kind CLAUDE.md rule 8c warns
  about (a later rewrite reintroduced `\operatorname` once already).
- **Verification performed (all passing).** `python3 experiments/check_render.py REPORT.md RESULTS.md`
  → **ALL CHECKS PASS**: REPORT.md 29 display equations, 449 inline equations, 27 embedded figures,
  0 problems; RESULTS.md 27 embedded figures, 0 problems. Every display equation renders as
  `js-display-math` on the GitHub API, no `<pre lang="math">` code blocks, no denylisted macros, and
  every inline `$…$` compiles after GitHub's backslash-stripping (rules 8a–8c).
- **Figure hygiene (rule 12) re-checked.** 27 `![…]` embeds and 27 visible `**Figure N.**` caption
  lines in each of REPORT.md and RESULTS.md (counts match exactly); zero bare `(plots/x.png)` paths
  outside an image embed; all 27 referenced PNGs exist on disk in `plots/`.
- **Structure (rule 8) re-checked.** REPORT.md is `Summary → Methods → Results → Conclusion`, with
  Methods giving data/model/layer and defining every metric and baseline with rendered equations.
- **Staleness sweep.** Grepped both deliverables for version-history language ("previously",
  "superseded", "v1/v2", "changed after review", "not performed", "was not run") — zero hits, so no
  superseded result or retracted claim survives in the curated files. The one remaining
  "sharper than the reference" sentence is the *fully-trained* framing (0.332 vs 0.351, p = 2.1e-4),
  which is a different comparison from the retracted matched-accuracy sub-claim and is hedged in its
  own text as "indistinguishable-to-slightly-sharper"; it is correct as written.
- **STOP written.** The plan (S1–S14) plus twelve PLAN-named follow-ups are complete and all five
  `human_feedback*` files end in `.addressed.md`, so zero unaddressed feedback remains (CLAUDE.md
  rule 11 satisfied).

## 2026-08-03 — S15: second seed at frozen-early; the depth step now separates with two seeds a side

Direction reopened by the wrapper (the previous `STOP` is gone). No unaddressed `human_feedback*` /
`*REVIEW*` file exists — all five end in `.addressed.md` — so this iteration advanced the plan by the
step the previous `Next step` named. The pod had also been reset, wiping every `/tmp` scratch
checkpoint and the corpus.

- **New experiment (`train_frozen.py --freeze 1,2,3,4 --seed 2024 --tag frozen_early_s2`, 30,000 steps,
  44.7 min).** Frozen-early retrained from a second model initialization, everything else identical
  (corpus SHA, split, data order, optimizer, cosine schedule, batch, checkpoint grid, freeze mask).
  Reaches the reference's final validation accuracy at step **2,750** — the same step as seed 1337 —
  with val 0.5529, and finishes at val **0.5629** (seed 1337: 0.5625). Scored on the same 150 pairs by
  `narrow_assay.py frozen_early_s2`, which now reads the frozen block list off the checkpoint so it
  handles frozen as well as narrow tags.
- **RESULTS.md / REPORT.md — seed-replication result added (new numbers, nothing superseded).**
  Frozen-early matched-accuracy median width **0.498** against seed 1337's 0.476, with per-pair widths
  statistically indistinguishable (paired `Δw` +0.001, exactly half the pairs each way, p = 0.40) and
  the two distributions agreeing decile by decile (10th/50th/90th 0.286/0.498/0.653 vs
  0.308/0.476/0.647). At step 30,000: **0.445** against 0.471, i.e. 0.027 in the opposite direction
  (paired −0.030, p = 3.3e-5). Seed noise is therefore ≤ 0.04 with no consistent sign. The relocation
  signature reproduces at both checkpoints (injection blocks 0/2/4 → 0.498/0.498/0.501 and
  0.445/0.444/0.443; the frozen group contributes nothing).
- **New cross-run statistic.** With two seeds at each end, the three runs with 12 trainable blocks
  (reference 0.443, narrow 0.397 and 0.437) are **disjoint** from the three with 8 (frozen-early 0.476
  and 0.498, frozen-late 0.500): one-sided rank-sum **p = 0.05**, the floor for a 3-vs-3 design.
  All four narrow-vs-frozen-early seed combinations agree pair by pair (−0.073, −0.067, −0.044,
  −0.063; each p ≤ 2.7e-8). `experiments/frozen_pairwise.py` now computes both the second-seed paired
  shifts and this rank test into `results/frozen_pairwise.json`.
- **CORRECTION carried into both deliverables — narrow-run parameter count.** The published count
  summed the model's `state_dict`, which double-counts the tied embedding/unembedding weight and adds
  the causal-mask buffers. Corrected to `model.parameters()`, the count used for every other run:
  **5,584,896 → 5,375,808**. The narrow run therefore has **4.0% fewer** trainable parameters than
  frozen-early's 5,601,360, not 0.3% more. This strengthens rather than weakens the depth conclusion —
  the sharper run is the one with *less* capacity — and the text now says so. Summary, Methods,
  Results and the Figure 24 caption updated in REPORT.md; the matching bullet and caption in RESULTS.md.
- **Figure 24 regenerated** (`plots/capacity_vs_depth.png`): six frozen-run markers instead of five
  (five frozen groups, two seeds of frozen 1–4), the corrected narrow-run x position, and each
  condition labelled once — the two-seed conditions read "(2 seeds)" with their two markers adjacent,
  which also fixed the label collisions the extra run introduced. Caption rewritten in both
  deliverables to define the new marker convention and to name both across-seed gaps.
- **Limitation 7 / caveats updated.** Two conditions now carry a second seed; the other four frozen
  conditions are still one seed each, so the *finer* ordering claim (5 trainable blocks near the
  readout beating 5 at the bottom, 0.558 vs 0.626) is flagged as resting on a gap only ≈1.5× the
  measured seed spread.
- **Rule 9a fix.** The "Models actually tested" table in RESULTS.md sat directly under its heading with
  no prose; a lead paragraph now states what each of the three models is doing and why it matters.
- **Repairs forced by the scratch wipe (no result changed).** `allpairs_sweep.load_vocab()` rebuilds
  the 65-character vocabulary from the SHA-verified corpus when the pilot checkpoint is missing
  (verified byte-identical vocabulary), and `plot_capacity.py` reads the narrow run's parameter count
  from `results/train_meta_narrow192.json` rather than loading a checkpoint.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 456 inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 embeds
  and 27 visible `**Figure N.**` captions per file; zero bare `(plots/x.png)` paths.
- **No `STOP` written** — wall clock and a named next experiment (a second seed at frozen-deep or
  frozen-mirror) both remain.

## 2026-08-03 — S16: second seed at frozen-deep; the position term now has an error bar under it

Direction still open (no `STOP` present). No unaddressed `human_feedback*` / `*REVIEW*` file exists —
all five end in `.addressed.md` — so this iteration advanced the plan by the step the previous
`Next step` named: the last sub-claim resting on a single pair of runs.

- **New experiment (`train_frozen.py --freeze 1,2,3,4,5,6,7 --seed 2024 --tag frozen_deep_s2`, 30,000
  steps, 20.9 min).** Frozen-deep retrained from a second model initialization, everything else
  identical (corpus SHA, split, data order, optimizer, cosine schedule, batch, checkpoint grid, freeze
  mask). Reaches the reference's final validation accuracy at step **3,000** — the same step as seed
  1337 — with val 0.5503, and finishes at val **0.5730** (seed 1337: 0.5742). Scored on the same 150
  pairs by `narrow_assay.py frozen_deep_s2`.
- **Prediction on record before scoring** (PLAN S16, REPORT Methods): the replicate lands within the
  ≈0.04 seed spread of frozen-deep's 0.558 and stays below frozen-mirror's 0.626. **CONFIRMED at both
  framings.**
- **RESULTS.md / REPORT.md — new numbers, nothing superseded.** Median width **0.559** at matched
  accuracy against seed 1337's 0.590, and **0.579** at step 30,000 against 0.558: within-condition
  spreads 0.031 and 0.021, with no consistent sign (paired `Δw` −0.016, p = 8.9e-4, then +0.023,
  p = 4.5e-5). Both seeds sit below frozen-mirror on both framings — worst seed still 0.039 (matched)
  and 0.046 (final) narrower — and pair by pair the replicate is **−0.060** vs frozen-mirror at matched
  accuracy (21% of pairs wider, p = 5.9e-14) and **−0.040** at the end of training (29%, p = 3.4e-8).
  Relocation signature reproduced (injection blocks 0/2/4 → 0.559/0.558/0.557 and 0.579/0.578/0.577;
  the frozen blocks 1–7 contribute nothing, all sharpening in 8–11). Geometry unchanged (`t*` 0.486,
  endpoints differ 88%, 3 argmax regions, |t*−t_flip| 0.084, partial ρ = −0.58, strict rate 0).
- **`experiments/frozen_pairwise.py`** gained the eight frozen-deep-second-seed comparisons and a
  `position_contrast_{matched,last}` summary (both seeds' medians, the seed spread, whether both sit
  below frozen-mirror, and the worst-seed gap) → `results/frozen_pairwise.json`.
- **Figure 24 regenerated** (`plots/capacity_vs_depth.png`): seven frozen-run markers instead of six,
  with frozen 1–7 now labelled "(2 seeds)" and its two markers adjacent; two label offsets moved so the
  frozen 1–7 and frozen 5–11 labels no longer collide on either panel. Caption rewritten in both
  deliverables to name all three across-seed spreads (0.397 vs 0.437 at 12 blocks, 0.476 vs 0.498 at 8,
  0.590 vs 0.559 at 5) and to state that both frozen-1–7 seeds sit below the frozen-5–11 diamond.
- **Stale closing prediction replaced in both hypothesis paragraphs.** They still ended on the narrow-run
  prediction, which S14 answered, and misstated the reference width as `n_embd` 384 (it is 240). They now
  report the narrow-run outcome and end on the open successor: five trainable blocks in the *middle*
  (freeze 0–3 and 9–11) should land near 0.58–0.60, between the two known five-block values.
- **Limitation 7 and the two closing paragraphs updated**: three conditions carry a second seed, and the
  position claim's remaining weakness is stated precisely — a seed spread exists on the frozen-deep side
  only, since frozen-mirror is still one run.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 472 inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 embeds and
  27 visible `**Figure N.**` captions per file; zero bare `(plots/x.png)` paths.
- **No `STOP` written** — wall clock and a named next experiment (the middle-of-stack five-block freeze)
  both remain.

## 2026-08-03 — S17: the middle-of-stack five-block freeze falsifies the count-first reading

Direction still open (no `STOP` present). No unaddressed `human_feedback*` / `*REVIEW*` file exists —
all five end in `.addressed.md` — so this iteration curated the run launched at the end of the previous
one and then launched its successor.

- **New experiment (`train_frozen.py --freeze 0,1,2,3,9,10,11 --tag frozen_mid`, 30,000 steps,
  20.1 min).** The same seven blocks frozen as frozen-deep and frozen-mirror (58.0% of parameters) and
  the same five trainable blocks, but in the *middle* of the stack (blocks 4–8) — the only one of the
  three five-block positions not yet run. Reaches the reference's final validation accuracy at step
  **3,750** (val 0.5519) and finishes at val **0.5728**. Scored on the same fixed 150 pairs by
  `narrow_assay.py frozen_mid`.
- **Prediction on record before scoring** (PLAN S17, REPORT Methods, and the hypothesis paragraphs of
  both deliverables): `w` lands between the two known five-block values, near **0.58–0.60**; at or
  below 0.559–0.590, or at or above 0.626, falsifies the ordered position reading. **FALSIFIED, and
  the falsification is the result.**
- **RESULTS.md / REPORT.md — new numbers.** `w` = **0.365** (IQR 0.253–0.471) at matched accuracy and
  **0.331** (IQR 0.258–0.428) at step 30,000 — below *both* positional siblings, below every
  eight-block run, and below the untouched 12-block reference at its own matched checkpoint (0.443;
  paired −0.056, 25% of pairs wider, p = 2.7e-14). Paired against the siblings: **−0.211** vs
  frozen-deep seed 1 (1.3% wider, p = 3.3e-26), **−0.188** vs seed 2 (0% wider, p = 2.3e-26),
  **−0.240** vs frozen-mirror (0.7%, p = 2.3e-26); at the end of training **−0.206** and **−0.260**,
  and **−0.023** against the fully trained reference (37% wider, p = 0.004). Strict plateau rate
  **24.7%** (matched) / 22.7% (final) against the reference's 10.0% and 0–0.7% for every other frozen
  run — the highest of any model in the report. Relocation signature holds (injection blocks 0/2/4/8 →
  0.331/0.342/0.525/0.802: 0.277 of the sharpening over blocks 5–8, 0.183 over 3–4 of which only block
  4 can train, 0.011 for the frozen blocks 1–2). Geometry ordinary (`t*` 0.501, endpoints differ 89%,
  3 argmax regions, |t*−t_flip| 0.048, partial ρ −0.47 matched / −0.61 final).
- **SUPERSEDED CLAIM (rule 9b re-framing).** Both deliverables previously read the frozen series as
  "trainable **depth** is the first-order term, position a second-order correction that favours blocks
  near the readout", supported by the monotone series 0.351 → 0.47–0.48 → 0.56–0.63 → 0.726 for 12, 8,
  5 and 1 usable trainable blocks. That is now wrong: **the three five-block runs span 0.365–0.629, a
  wider range than the entire 12-to-5 series**, and five mid-stack blocks beat all three eight-block
  runs (0.476–0.500) and the 12-block reference. The current reading, in both files, is that
  **position** governs the cost, that it is non-monotone with an interior optimum (the two-point
  contrast sampled only the ends), and that the three points are described by how the seven frozen
  blocks are *distributed* around the trainable window — mid splits them 3/3, deep stacks all seven
  before it, mirror all seven after it (0.365 < 0.558/0.590 < 0.626/0.629) — labelled explicitly as a
  description of three points rather than a tested law. Rewritten in the Summary, the hypothesis
  paragraphs, the frozen-block Results section, "What this settles", the Conclusion and Limitation 7 of
  REPORT.md, and the matching passages of RESULTS.md.
- **Other claims corrected as a consequence.** "The sharpest tail is what every frozen run loses"
  (0.7% early, 0% the rest against the reference's 10%) → five of six lose it and frozen-mid gains it
  at 24.7%. "Trainable depth, not trainable capacity, is the variable that matters" → "Trainable
  capacity is not the variable at all" (the narrow run's evidence is unchanged; only the positive half
  of the claim was withdrawn). The relocation paragraph now reports six sites, not five, and the
  geometry paragraph five plateau-retaining runs, not four.
- **Figures.** `plots/frozen_blocks.png` (Figure 23) regenerated with the frozen-mid condition: a ninth
  curve panel, two more width points, and a seventh series in the depth and accuracy panels — which
  needed a second gray lightness (`0.62`) plus a new dash pattern and marker, since the CVD palette
  holds five hues and the rule forbids a sixth. `plots/capacity_vs_depth.png` (Figure 24) redrawn with
  the new diamond at 5 trainable blocks; both panel titles and the suptitle rewritten, because "ordered
  by trainable DEPTH" is no longer true of the left panel. Captions in both deliverables rewritten to
  state what each panel now shows.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 501 inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 embeds
  and 27 visible `**Figure N.**` captions per file; zero bare `(plots/x.png)` paths.
- **Successor launched in the same iteration with its prediction pre-registered** (PLAN S18):
  `--freeze 0,1,2,3,4,8,9,10,11 --tag frozen_mid3`, a *three*-block mid-stack window (blocks 5–7, 74.6%
  of parameters frozen). Predicted 0.40–0.50 if position dominates the count; at or above 0.558 the
  count returns as the leading term.
- **No `STOP` written** — wall clock and a named running experiment both remain.

## 2026-08-03 — S18: three mid-stack blocks match the full network (position confirmed as the variable)

Same iteration as S17 above. Launched immediately after S17's falsification, with its prediction
written into PLAN.md and REPORT.md Methods before it was scored.

- **New experiment (`train_frozen.py --freeze 0,1,2,3,4,8,9,10,11 --tag frozen_mid3`, 30,000 steps,
  19.4 min).** Nine of twelve blocks frozen (**74.6%** of parameters), leaving a *three*-block trainable
  window at the same mid-stack site S17 found (blocks 5–7). Its purpose is to separate the trainable
  window's **position** from its **size**. Reaches the reference's final validation accuracy at step
  **7,000** (val 0.5518) and finishes at val **0.5711**.
- **Prediction on record before scoring:** ≈**0.40–0.50**, still beating five blocks at either end, if
  mid-stack position dominates window size; **≥ 0.558** if the block count reasserts itself once the
  window shrinks. **CONFIRMED.**
- **RESULTS.md / REPORT.md — new numbers.** `w` = **0.446** (IQR 0.344–0.559) at matched accuracy and
  **0.427** (IQR 0.324–0.541) at step 30,000. Statistically **indistinguishable from the full 12-block
  reference** at the matched checkpoint (0.443; paired +0.009, 55% of pairs wider, p = 0.17), and
  clearly sharper than every five-block window at an end: **−0.121** vs frozen-deep seed 1 (9.3% of
  pairs wider, p = 7.2e-23), **−0.090** vs seed 2 (11%, p = 1.4e-21), **−0.154** vs frozen-mirror
  (4.7%, p = 1.3e-25); at the end of training **−0.111** and **−0.184**. Strict plateau rate 9.3%
  (matched) / 10.0% (final) against the reference's 12.7% / 10.0%. Depth profile
  0.427/0.436/0.443/0.806/0.806/0.806 — 0.363 of its total 0.380 between injection blocks 4 and 8, the
  tightest relocation in the series; frozen blocks 1–4 and 9–11 contribute 0.017 between them. Geometry
  ordinary (`t*` 0.479, endpoints differ 91%, 3 argmax regions, |t*−t_flip| 0.079, partial ρ −0.46).
- **What it settles, and what it costs.** Window *size* is not irrelevant — five mid-stack blocks to
  three costs +0.086 of width (85% of pairs wider, p = 3.0e-17) and the run needs 7,000 steps rather
  than 3,750 to reach the reference's accuracy — but position dominates it: three trainable blocks in
  the middle beat five at either end by 0.09–0.18 and reproduce a 12-block network's plateau geometry
  with nine of twelve blocks never leaving their initialization. This is the first successful test of
  the "long adjacent frozen stretch is what costs sharpness" reading S17 could only describe.
- **Claims updated as a consequence.** Run counts throughout both deliverables (seven frozen groups,
  eight retraining runs, six that retain a plateau); the strict-tail sentence now names both mid-stack
  runs as exceptions; the narrow run is no longer "the only non-reference run that keeps the sharpest
  tail" (13.3% against frozen-mid's 24.7% and frozen-mid3's 9.3%) — corrected in both files; frozen-two
  is wider than the untrained network on 26% of pairs against 0–1.3% (was 0–1%) in the other six runs.
  Both hypothesis paragraphs now end on a new open prediction: a five-block window one step off-centre
  (freeze 0–1 and 7–11, leaving blocks 2–6) should land between 0.365 and 0.558, nearer the former.
- **Figures.** `plots/frozen_blocks.png` (Figure 23) gained a tenth curve panel, two more width points
  and an eighth series; the styling was reworked so the five CVD hues go to the reference and the four
  runs carrying the position result, with the other three in three gray lightnesses, each with its own
  dash pattern and marker. The accuracy panel's seven per-run legend entries collapsed into one, since
  the steps are now given in the caption. `plots/capacity_vs_depth.png` (Figure 24) gained the
  three-block run at x = 3 with a label offset so it clears the frozen-5–11 label. Captions rewritten
  in both deliverables to match the new styles and to name each run's matched-accuracy step.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 536 inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 embeds
  and 27 visible `**Figure N.**` captions per file; zero bare `(plots/x.png)` paths.
- **No `STOP` written** — wall clock and named next experiments both remain.

## 2026-08-03 — S19: the off-centre window falsifies the "adjacent frozen stretch" reading

Same iteration as S17 and S18 above. Launched after S18 landed, with its prediction written into
PLAN.md and REPORT.md Methods before it was scored.

- **New experiment (`train_frozen.py --freeze 0,1,7,8,9,10,11 --tag frozen_mid_off`, 30,000 steps,
  19.6 min).** Seven blocks frozen (**58.0%** of parameters — the same fraction and the same five
  trainable blocks as frozen-deep, frozen-mirror and frozen-mid), but the window slid one step off
  centre to blocks 2–6, so five frozen blocks sit below it instead of three. Matches the reference's
  accuracy at step **3,500** (val 0.5507); finishes at val **0.5744**, the highest of any run in the
  study.
- **Prediction on record before scoring:** ≈**0.40–0.45**, between frozen-mid's 0.365 and frozen-deep's
  0.558, if the cost tracks how the frozen blocks are distributed around the trainable window; anything
  at or below 0.365 counts against that description. **FALSIFIED.**
- **RESULTS.md / REPORT.md — new numbers.** `w` = **0.365** (IQR 0.271–0.468) at matched accuracy and
  **0.355** (IQR 0.275–0.405) at step 30,000 — **identical to frozen-mid** on both framings (paired
  +0.014, p = 0.064; +0.007, p = 0.229; neither significant), sharper than the reference at its matched
  checkpoint (−0.050, 25% of pairs wider, p = 4.2e-12) and level with the fully trained reference
  (−0.009, p = 0.286). Against the two end windows: **−0.197** vs frozen-deep seed 1, **−0.172** vs
  seed 2, **−0.225** vs frozen-mirror at matched accuracy, and −0.200 / −0.264 at the end (all
  p ≤ 3.3e-25). Strict plateau rate **21.3%** / 18.7%. Depth profile
  0.355/0.525/0.737/0.807/0.807/0.807 — 0.382 of its 0.452 total across blocks 1–4. Geometry ordinary
  (`t*` 0.488, endpoints differ 92%, 3 argmax regions, |t*−t_flip| 0.062, partial ρ −0.53).
- **SUPERSEDED CLAIM.** The description added one entry above — that the cost tracks how the seven
  frozen blocks are *distributed* around the trainable window (frozen-mid splits them 3/3, frozen-deep
  stacks seven before, frozen-mirror seven after) — is **withdrawn from both deliverables**: doubling
  the frozen stretch below the window (three → five) changed nothing at all.
- **REPLACEMENT CLAIM, stated with its caveats.** Ordering the eight frozen runs by their *usable*
  window (trainable blocks intersected with 1–11, since patching at block 0 overwrites block 0's
  output) separates them exactly. **Strictly interior** windows — blocks 4–8, 2–6, 5–7 — give 0.365,
  0.365, 0.446. Windows **touching an end** — 5–11, 1–7, 8–11, 1–4, block 11 alone — give 0.476, 0.500,
  0.590, 0.629, 0.712. No overlap. Both deliverables state that this was found after the fact rather
  than predicted, that its mechanism is not established here, and that while the two five-block
  interior windows clear every end-touching run by ≥ 0.11 (several times the ≈0.04 seed spread), the
  three-block interior window sits only 0.030 below the nearest end-touching one, inside that spread.
- **New pre-registered prediction in both hypothesis paragraphs and the Conclusion:** a five-block
  window at blocks 1–5 — one block down from the sharp 2–6 window, hence touching the block immediately
  after the patched activation — must land above 0.47 with the blunt group; near 0.365 would make the
  interior/end split a coincidence of these eight runs.
- **Claims updated as a consequence.** Eight frozen groups and nine retraining runs throughout; seven
  runs retain a plateau; three interior-window runs keep the sharpest tail (24.7%, 21.3%, 9.3%);
  frozen-two's "wider than untrained" contrast is now against the other seven runs; the four-way
  five-block comparison replaces the three-way one as "the cleanest in the series".
- **Figures.** `plots/frozen_blocks.png` (Figure 23): eleven curve panels, and the depth panel **split
  into two small multiples** of five series each (the four five-block windows plus the reference; the
  other freeze sizes plus the reference) — nine series in one panel exceeded the five-hue palette, and
  CLAUDE.md rule 13 asks for small multiples above five categories. The trained reference moved to
  neutral black as the anchor, freeing the five CVD hues for the runs that carry the position result,
  with the remaining three in gray at three lightnesses; every series keeps its own dash pattern and
  marker. `plots/capacity_vs_depth.png` (Figure 24): the fourth five-block diamond, new nudge offsets
  so all four are separable, and the legend moved to the upper left where the new marker needed room.
  Both captions rewritten in both deliverables.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 558 inline equations / 27 figures; RESULTS 27 figures; 0 problems); 27 embeds
  and 27 visible `**Figure N.**` captions per file; zero bare `(plots/x.png)` paths.
- **No `STOP` written** — a named, pre-registered next experiment remains.

## 2026-08-03 — S20: the pre-registered interior/end test runs, and REFUTES the split

- **New run (ninth frozen group, fourteenth model): `frozen_mid_low`** — `train_frozen.py --freeze
  0,6,7,8,9,10,11 --tag frozen_mid_low`, five trainable blocks at 1–5, 58.0% of parameters frozen,
  everything else identical to the reference character run. Matched the reference validation accuracy
  0.5502 at step 3,500 (val 0.5503); finished step 30,000 at val 0.5732.
- **The pre-registered prediction FAILED, and both deliverables now say so.** The interior/end split
  (recorded in PLAN, both hypothesis paragraphs and the Conclusion last iteration) required this window
  to land **above 0.47** because its usable window touches block 1. It lands at **0.363** (IQR
  0.280–0.447) at matched accuracy and **0.326** (IQR 0.258–0.416) at step 30,000 — the sharpest final
  width of any model in the study — indistinguishable from the two mid-stack five-block windows
  (paired Δw +0.008, p = 0.27; −0.009, p = 0.23) and 0.10–0.23 sharper than every end window
  (p ≤ 2e-21). **The interior/end split is withdrawn from both deliverables** (old: "strictly interior
  windows 0.365–0.446, end-touching 0.476–0.712, no overlap, pre-registered test waiting" → new: the
  test was run and refuted it).
- **New rule-free claim promoted to the front of the frozen-block story.** frozen_mid_low's trainable
  blocks 1–5 are a strict **subset** of frozen-late's trainable 0–7, and it is **0.118 sharper**
  (4.7% of pairs wider, p = 2.2e-25). Removing two trainable blocks sharpens the plateau — a direct
  two-network comparison that needs no regularity fitted to nine runs. This now leads the Summary and
  the "what this settles" paragraph in both files, replacing the geometric split.
- **Replacement description, explicitly given no predictive credit.** Every usable window containing
  mid-stack **block 5** gives 0.363, 0.365, 0.365, 0.446, 0.476, 0.500; the three excluding it (8–11,
  1–4, block 11) give 0.559–0.590, 0.629, 0.712. Vivid pair: blocks 1–4 (0.629) vs blocks 1–5 (0.363).
  Both deliverables state that the previous description died on its first test, so this one earns
  nothing until it survives the same treatment.
- **New pre-registered prediction (replaces the refuted one) in PLAN, both hypothesis paragraphs, the
  Results section and the Conclusion:** a five-block window at blocks **6–10** (freeze 0–5 and 11)
  excludes block 5 while touching neither end, so the coverage description requires **≥ 0.55** — on a
  window the refuted rule would have called sharp. Near 0.365 refutes it in turn.
- **Counts updated throughout both files:** nine frozen groups / eleven frozen runs / fourteen models;
  five runs at exactly five trainable blocks spanning 0.363–0.629; four mid-stack-window runs keep the
  strict tail (24.7%, 21.3%, **19.3%**, 9.3%) against five that lose it.
- **Figures.** `plots/frozen_blocks.png` (Figure 23): twelve curve panels; the ten-series palette
  re-assigned along the two small multiples — the five CVD hues now go to the five five-block windows
  (left panel, the runs carrying the position result) and the four other freeze sizes to gray at four
  lightnesses (right panel), each series keeping its own dash pattern and marker.
  `plots/capacity_vs_depth.png` (Figure 24): a fifth diamond in the 5-block column on a finer nudge
  grid. Both captions rewritten in both deliverables (panel lists, sharpest panel now the tenth,
  matched-accuracy step list).
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 569 inline equations / 27 figures; RESULTS 27 figures; 0 problems).
- **No `STOP` written** — a named, pre-registered next experiment remains.

## 2026-08-03 — S21: the pre-registered coverage test runs, REFUTES it, and gives the sharpest network

- **New run (tenth frozen group, fifteenth model): `frozen_high`** — `train_frozen.py --freeze
  0,1,2,3,4,5,11 --tag frozen_high`, five trainable blocks at 6–10, 58.0% of parameters frozen at their
  random initialization, everything else identical to the reference character run. Reached the
  reference's validation accuracy 0.5502 at step 3,750 (val 0.5523).
- **The pre-registered prediction FAILED, and both deliverables now say so.** The coverage description
  (recorded in PLAN, both hypothesis paragraphs, Results and the Conclusion last iteration) required
  this window to land **at or above 0.55** because its usable window excludes mid-stack block 5. It
  lands at **0.342** (IQR 0.240–0.446) — the **sharpest matched-accuracy width of the fifteen models in
  the study**, sharper than the two mid-stack windows it was predicted to lose to by 0.19 (paired Δw
  −0.014, p = 0.025 vs blocks 4–8; −0.024, p = 9e-4 vs blocks 1–5) and 0.14–0.25 sharper than every end
  window (p ≤ 1.2e-25). Strict plateau rate **28.0%**, the highest of any condition measured (old
  highest: 24.7%, frozen-mid). **Coverage is withdrawn from both deliverables** (old: "every window
  containing block 5 gives 0.363–0.500, the three without it 0.559–0.712, test pending" → new: the test
  was run and refuted it).
- **Both deliverables now state plainly that ten runs support NO geometric summary.** Two post-hoc
  descriptions (interior/end, then coverage) have each died on the first experiment aimed at it, so
  neither trainable count, trainable capacity nor window geometry predicts the width.
- **Second rule-free claim promoted alongside the ninth run's subset comparison.** Blocks 6–10 alone,
  with 58.0% of the network never moved from initialization, are **0.072 sharper than the untouched
  12-block reference** at the same validation accuracy (18.7% of pairs wider, p = 8.5e-18). Training
  fewer blocks sharpened the plateau — a second direct two-network fact that needs no fitted rule.
- **Counts updated throughout both files:** ten frozen groups / twelve frozen runs / fifteen models;
  five of ten frozen runs lose the strict tail, the five mid-stack-window runs keep it (**28.0%**,
  24.7%, 21.3%, 19.3%, 9.3%).
- **Next tests are now seed replications, not a third fitted rule** — a second seed at frozen-mirror
  (the one single-seed run carrying a load-bearing comparison) and a second seed at blocks 6–10 to
  confirm the study's sharpest network is not a seed artefact. Both deliverables and PLAN say so.
- **Figures.** `plots/frozen_blocks.png` (Figure 23): the depth small multiples went from two panels to
  **three** — six five-block runs no longer fit two panels inside the five-hue palette, so they are
  split by where the trainable window sits (upper-stack windows 6–10, 4–8, 2–6; bottom windows 1–5, 0–4
  plus the non-window trainable set 0&8–11), with the other freeze sizes in the third panel in gray at
  four lightnesses; no panel now carries more than three hues plus the black reference anchor.
  `plots/capacity_vs_depth.png` (Figure 24): a sixth diamond in the 5-block column on a seven-slot
  nudge grid. Both captions rewritten in both deliverables.
- **One scope note recorded in the deliverables:** the step-30,000 assay of `frozen_high` was still
  training when the iteration closed, so this run enters the deliverables at its matched-accuracy
  checkpoint only — which is the primary comparison axis throughout this section. Both Figure 23
  captions state it.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 577 inline equations / 27 figures; RESULTS 27 figures; 0 problems).
- **No `STOP` written** — the run's final-step assay and two named seed replications remain.

## 2026-08-03 — S22: the sharpest network's final-step row lands, and the comparison survives it

- **No new training.** The chained `narrow_assay.py frozen_high` from S21 completed (trainer reached
  step 30,000, val_acc 0.5720, best 0.5764) and wrote `frozen_high_last` into
  `results/frozen_assay_summary.json`. This iteration picks up that row.
- **The step-30,000 numbers strengthen the claim they were pending on.** Blocks 6–10 trainable
  (58.0% of parameters frozen at random init) give median `w` **0.328** (IQR 0.252–0.395, strict
  plateau rate **24.0%**) at step 30,000, against the untouched 12-block reference's 0.351 at the
  same step — paired **−0.037**, 36.7% of pairs wider. Both networks have now run the full schedule,
  so the "training fewer blocks can sharpen the plateau" fact no longer rests only on the
  matched-accuracy checkpoint.
- **The run kept sharpening past matched accuracy** (0.342 at step 3,750 → 0.328 at step 30,000),
  which makes the matched-accuracy comparison the conservative one. Both deliverables say so.
- **Scope note retired from both deliverables** (old: Figure 23's caption said the eleventh frozen
  group's "step-30000 assay was still running when this iteration closed" → new: the caption states
  the figure shows the matched-accuracy checkpoint because that is the section's primary comparison
  axis, and points to the step-30,000 width in the text).
- **Figures unchanged this iteration** — Figure 23 continues to show `frozen_high` at its
  matched-accuracy checkpoint; no plot was regenerated, and no claim in either deliverable depends on
  a figure that does not exist.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 581 inline equations / 27 figures; RESULTS 27 figures; 0 problems).
- **No `STOP` written** — two named seed replications remain.

## 2026-08-03 — FINALIZATION: deliverables verified current-best, `STOP` written

- **No content change to RESULTS.md or REPORT.md.** Both were re-read in full and confirmed to already
  reflect the current-best state as curated at S22. No experiment ran this iteration, nothing was
  running at close, and every row in `results/` (through `frozen_assay_summary.json`, 21:02) is already
  in both deliverables — so there was no superseded number to replace and no old → new to record.
- **Why no new experiment.** ~14 minutes of wall clock remained. The next planned step (S23) is a seed
  replication needing ~21 min of training plus ~70 s of assay; starting it would have produced an
  unusable partial checkpoint and no deliverable row. Finalization was the correct use of the time.
- **The two seed replications stay in both deliverables as a stated limitation**, not as pending work:
  a second seed at blocks 6–10 and one at frozen-mirror. Three of the frozen conditions already carry a
  second seed and the measured seed spread there is small (0.031 at matched accuracy), which is the
  caveat the reader needs to size the risk.
- **Feedback status.** Five `human_feedback*` files, all five ending in `.addressed.md`. **Zero
  unaddressed feedback**, which is what permits `STOP` under CLAUDE.md rule 11.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 29 display / 581 inline equations / 27 embedded figures; RESULTS 27 figures; 0 problems).
  Figure embeds 27 and visible `**Figure` captions 27 in each file; `grep -nE '\(plots/[^)]+\.png\)'`
  returns no un-embedded bare path in either file.
- **`STOP` written** — plan complete for the available budget and zero unaddressed feedback.

## 2026-08-10 — operator feedback #5 addressed: new companion deliverable `REPORT_followup.md`

- **Feedback file.** `human_feedback_5.txt` (four asks, numbered 1/2/4/6 by the operator) was the whole
  of this iteration; it is now `human_feedback_5.txt.addressed.md`. The operator asked explicitly that
  the answers go into a **new** file, `REPORT_followup.md`, written under the same rules as REPORT.md —
  so REPORT.md and RESULTS.md keep their existing current-best content and gain only a pointer
  paragraph each (see below). No previously reported number changed; nothing was superseded.
- **New deliverable `REPORT_followup.md`** (Summary → Methods → Results → Conclusion; 4 display
  equations, 4 embedded figures, each with a visible numbered caption). New numbers it reports, all
  from re-analysis of the stored 2,080-pair sweep — **no training and no new forward pass**:
  - item 1 — pairwise width matrix with undertrained characters (< 1000 occurrences in the training
    split) removed: 12 characters dropped (`$ & 3 X Z Q J z x q j V`), 53 kept. Median
    $w_{10\to90}$ **0.355 over all 2,080 pairs → 0.320 over the 1,378 well-trained pairs**, against
    **0.482** for the 702 pairs touching an undertrained character (one-sided Mann–Whitney
    p = 4.0e-159); strict-plateau rate **8.8% → 11.8%**. Trend: per-character median width vs
    log training frequency, Spearman **ρ = −0.78** (p = 1.0e-14, n = 65) and **ρ = −0.66**
    (p = 6.9e-8) within the 53 kept characters.
  - item 2 — example `d(t)` curves for six pairs with both endpoints well trained, each drawn against
    its own mirror image; no metrics reported, per the request. Recorded alongside them: the endpoint
    *order* is exactly symmetric (stored 100-pair swap control, max |w(A,B) − w(B,A)| = 0.000), so the
    live question is curve shape, which is markedly asymmetric and follows endpoint plausibility.
  - item 4 — context/sample-count audit. Every character-level interpolation figure uses a real prompt
    context (all-pairs, comma and intervention figures: `"The house was "`, 14 characters, final
    position patched; context-control figures: 9 contexts). A table in Methods gives the context and
    the per-cell sample count for every character-level figure of REPORT.md.
  - item 6 — widths from the letter `e` to all 52 other well-trained characters, grouped by character
    class: groups differ (Kruskal–Wallis p = 9.2e-3), and the same group ordering (lower-case vowels
    0.270 < whitespace 0.286 < upper-case vowels 0.316 < lower-case consonants 0.320 < punctuation
    0.331 < upper-case consonants 0.356) repeats across all **43** well-trained letters (Friedman
    p = 3.8e-18, Kendall **W = 0.42**), surviving removal of the partner-frequency confound
    (**W = 0.27**, p = 4.3e-11).
- **New figures** (all CVD-safe, hue + hatch/marker, no red-green contrast), embedded in
  REPORT_followup.md as rendered images with visible captions: `plots/followup_width_matrix_trained.png`
  (Figure 1), `plots/followup_width_vs_freq.png` (Figure 2), `plots/followup_asymmetry_examples.png`
  (Figure 3), `plots/followup_letter_groups.png` (Figure 4).
- **Six existing REPORT.md figures re-rendered** (Figures 14–19, `plots/allpairs_*.png`) with one added
  footer line each stating the prompt context and the sample count behind a cell/point — feedback item
  4's "state that in the plot". No data, axis, series or caption changed, so the existing captions and
  numbers in REPORT.md and RESULTS.md remain exact.
- **Pointer paragraphs added** to REPORT.md (end of Summary) and RESULTS.md (head of the all-pairs
  section) naming `REPORT_followup.md` and its three findings, so neither deliverable hides the
  follow-up from a reader.
- **New code:** `experiments/followup_freq.py` (frequency + group analysis and all four figures) and
  `experiments/followup_asym.py` (both-direction re-run of six pairs; kept for reproducibility but not
  runnable this iteration — see JOURNAL). New data: `results/followup_summary.json`,
  `results/char_freq.json`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md REPORT_followup.md` →
  **ALL CHECKS PASS** (REPORT 29 display / 584 inline / 27 figures; RESULTS 27 figures;
  REPORT_followup 4 display / 60 inline / 4 figures; 0 problems). 4 embeds and 4 visible `**Figure`
  caption lines in the new file.
- **No `STOP`**: the iteration ended with the plan's own next step (the two seed replications) still
  open.

## 2026-08-10 — S23: the last two single-seed runs replicated; new Figure 25 (seed replication)

- **Two new training runs** (`experiments/train_frozen.py`, model seed 2024, everything else identical
  to their seed-1337 originals — corpus, split, data order, optimizer, cosine schedule, batch size,
  checkpoint grid, freeze mask), scored by `experiments/narrow_assay.py` on the same 150 character
  pairs at interpolation block 0: `frozen_high_s2` (freeze 0–5 and 11, i.e. blocks 6–10 trainable) and
  `frozen_mirror_s2` (freeze 5–11, i.e. blocks 0–4 trainable). Both predictions were pre-registered in
  PLAN.md while the runs were still training, and **both held**.
- **New numbers in RESULTS.md and REPORT.md** (nothing superseded — these are new rows):
  - blocks 6–10, seed 2024: median `w` **0.344** (IQR 0.278–0.418) at matched accuracy (step 3,750,
    val 0.5530) against seed 1337's **0.342** — across-seed spread **0.002**, the smallest in the study
    (paired Δw +0.007, 53% of pairs wider, p = 0.65) — and **0.335** against **0.328** at step 30,000
    (spread 0.007, p = 0.068). Against the untouched 12-block reference: **−0.071** at matched accuracy
    (18.0% of pairs wider, p = 1.9e-16) and **−0.021** at step 30,000 (37.3%, p = 7.5e-4). Strict
    plateau rate 18.0% / 20.7%; relocation and geometry reproduce the first seed.
  - blocks 0–4, seed 2024: **0.624** at matched accuracy (step 2,750, val 0.5504) against **0.629**
    (spread 0.006, p = 0.35) and **0.590** against **0.626** at step 30,000 (spread 0.036, p = 1.2e-4).
    All four deep-versus-mirror seed pairings preserve the ordering: **+0.031** (p = 3.4e-10) and
    **+0.053** (p = 1.8e-16) at matched accuracy, **+0.038** (p = 6.0e-9) and **+0.022** (p = 3.1e-3)
    at step 30,000.
  - across-seed spread over the five twice-trained conditions and nine condition-by-checkpoint pairs:
    **0.002–0.040** (max = the narrow run's 0.397 vs 0.437 at matched accuracy), sharper on the second
    seed in four of nine pairs and blunter in five.
- **Claim narrowed (statement changed, no number retracted).** The near-readout-versus-bottom position
  ordering was previously reported with the two frozen-deep seeds below frozen-mirror's single run.
  With two seeds a side its direction survives at both checkpoints, but its closest median pairing is
  **0.033** at matched accuracy and **0.010** at the end of training — at or inside the 0.040 seed
  spread — so both deliverables now state that this ordering rests on the paired per-pair tests rather
  than on a median gap that clears seed noise by itself. Limitation 7 in REPORT.md rewritten
  accordingly (old: "the ordering of the two ends rests on one run each"; also corrected there: the
  frozen-block series uses **ten** frozen groups, not eight, and **four** of them now carry a second
  seed).
- **New figure, embedded with a visible caption in BOTH deliverables:** `plots/seed_replication.png`
  (**Figure 25**, `experiments/plot_seeds.py`) — left panel, the two seeds of each twice-trained
  condition at matched accuracy and at step 30,000; right panel, the six load-bearing gaps against the
  largest measured seed spread, with the two that do not exceed it hatched. Figure numbering is now
  contiguous 1–28 in both files (the exploratory figures had already been renumbered 26–28 in
  anticipation of this insert).
- **Figure 24 (`plots/capacity_vs_depth.png`) re-rendered** with the two new second-seed markers
  (fourteen frozen runs now, up from twelve) and with the six five-block conditions' labels moved into
  free space on leader lines — nine markers now share that column and the labels no longer fit beside
  them. Its caption updated: fourteen frozen runs, two seeds each of frozen 1–4, 1–7, 0–5&11 and 5–11,
  and the seed spreads at x = 5 now read 0.590 vs 0.559, 0.342 vs 0.344 and 0.629 vs 0.624.
- **Prose updated where it named these runs as future work:** RESULTS.md's "What this settles" no
  longer points at the two replications as the next runs (they are done); REPORT.md's Summary now says
  five conditions carry a second seed and bounds the spread at 0.040; REPORT.md's Conclusion replaces
  the "next tests would be seed replications" sentence with what the replications found and what
  remains open (the mechanism).
- **Code:** `experiments/plot_seeds.py` (new figure; legend and label placement fixed so no marker is
  covered), `experiments/frozen_pairwise.py` (+1 comparison: the tightest cell of the position
  contrast, `frozen_mirror_s2_last` vs `frozen_deep_s2_last`), `experiments/plot_capacity.py` (leader
  lines). New data: `results/frozen_assay_summary.json` and `results/frozen_assay_raw.npz` gained four
  rows; `results/frozen_pairwise.json` regenerated; `results/train_{hist,meta}_frozen_{high,mirror}_s2.json`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md REPORT_followup.md` →
  **ALL CHECKS PASS** (REPORT 29 display / 602 inline equations / 28 embedded figures; RESULTS 28
  figures; REPORT_followup 4 display / 60 inline / 4 figures; 0 problems).
- **No `STOP`**: zero unaddressed feedback, but the plan's remaining candidates (the mechanism probe,
  the PLAN 5.5 freeze-and-retrain prediction) are open.

## 2026-08-10 — finalization pass (no new experiment; budget exhausted)

- **No content change to RESULTS.md or REPORT.md.** Both were curated to current-best in the S23
  iteration earlier today (fourteen frozen runs, five conditions with a second seed, seed spread
  bounded at 0.040) and re-verified there; nothing in this pass supersedes a number, so per rule 6
  the deliverables were left exactly as they stand rather than churned.
- **Deliverable state at close:** REPORT.md — Summary → Methods → Results → Conclusion, 29 display
  equations, 602 inline, 28 embedded figures each with a visible bold `**Figure N.**` caption line and
  a prose citation by number; RESULTS.md — the same 28 figures, current-best only; REPORT_followup.md —
  4 display / 60 inline / 4 figures. Last `experiments/check_render.py REPORT.md RESULTS.md
  REPORT_followup.md` run: **ALL CHECKS PASS, 0 problems** (rules 8a–8d, 9a, 9d, 12).
- **Feedback:** zero unaddressed files — all six `human_feedback*` are `.addressed.md`.
- **`STOP` written** this iteration: wall-clock budget is exhausted and no unaddressed feedback file
  exists (rule 11). Open work is recorded in PLAN.md "Next step" (S24: probe what the trainable blocks
  compute; non-final-token interpolation; a second model) for whoever picks this up.

## 2026-08-12 — operator feedback #6: the basin criterion was redefined and validated against nulls

- **Feedback addressed:** `human_feedback_6.txt` → `human_feedback_6.txt.addressed.md`. The operator
  was right: the old basin fraction fired on the linear null. It required $t_{lo}\ge0.10$ (as $A$) or
  $t_{hi}\le0.90$ (as $B$), and $d(t)=t$ gives *exactly* $t_{lo}=0.10$, $t_{hi}=0.90$ — the threshold
  sat on the null, so the statistic could not fail and "all 65 characters own a basin" was unsupported.
  The transition-width results are unaffected, as the operator anticipated: the strict plateau rule
  also demands $w\le0.25$ and the line scores 0.80, so every width, strict-count, variance-decomposition,
  intervention and frozen-run number in both deliverables is unchanged.
- **New definition (REPORT.md Methods).** Rest length $r_c(\delta)$ = the path fraction staying within
  $\delta$ of $c$'s output, read at whichever end $c$ occupies; **rest ratio** $R_c=r_c(\delta)/\delta$,
  which is identically 1 for $d(t)=t$ at every $\delta$; basin iff $R\ge\kappa$ with $\delta=0.10$,
  $\kappa=2$ (the path parks on $c$ at least twice as long as a uniform morph would).
- **New baseline (REPORT.md Methods §Baselines).** Null-curve baseline: exact line; line + Gaussian
  noise ($\sigma$ = 0.01/0.02/0.05, 2,000 draws each); the untrained network's 2,080 curves; the
  200-pair block-11 patch. False-positive rate at $\kappa=2$: **0/2, 0/12,000, 0/4,160, 0/400**
  endpoint decisions. Under the old criterion those same families passed at **50%, ~50%, 40.8%,
  36.8%**. Trained network: **90.3%** of 4,160 endpoints, median rest ratio **3.18** (nulls 0.94–1.00).
- **Superseded headline number (old → new).** "Every character owns a basin, $\phi\ge0.86$ for all
  65, median 1.00" → "**59 / 65** characters hold a basin against a majority of partners (median
  $\phi$ = 1.00, mean 0.90; 55 at $\phi\ge0.9$, 39 at $\phi=1$); six fail — `3` 0.03, `&` 0.16,
  `$` 0.25, `Z` 0.31, `X` 0.47, `z` 0.47". Changed in RESULTS.md and REPORT.md in the Summary, the
  all-pairs table, the per-character verdict, the hypothesis statement and the Conclusion.
- **New supporting result.** The six failures are the six rarest characters: $\phi$ vs training-set
  frequency, Spearman $\rho=0.56$, $p=1.0\times10^{-6}$, n = 65; every character seen $\ge$ 1,000
  times has $\phi\ge0.68$ (53 characters). This is why the narrowed claim is stronger than the old
  one — the criterion now discriminates, and what it discriminates by is interpretable.
- **New caveat (REPORT.md, now item 3 of 8; later items renumbered).** Basin ownership is a majority
  claim resting on $\kappa$ and $\delta$: at $\delta=0.20$ the trained pass rate falls 90.3% → 52.0%
  (untrained stays 0.0%). The null separation and the character ordering do not depend on the threshold.
- **Figures.** Two new: **Figure 15** `plots/basin_criterion.png` (rest-length definition; pass rate
  vs $\kappa$ for trained and all null families; per-character $\phi$ with the untrained control and
  the old criterion overlaid) and **Figure 17** `plots/basin_vs_frequency.png` ($\phi$ vs training
  frequency). `plots/allpairs_width_by_char.png` (Figure 16) regenerated with the validated $\phi$ on
  its right axis, replacing the old `flat_frac`; its title and the small-multiples titles updated.
  Figures 16–28 renumbered to 18–30 in both deliverables, with every prose citation and range updated;
  30 embeds / 30 captions / sequential numbering in each file.
- **Code:** new `experiments/basin_criterion.py` (criterion + null validation + per-character table)
  and `experiments/plot_basin_criterion.py`; `experiments/plot_allpairs.py` reads $\phi$ from the new
  JSON. New data: `results/basin_criterion.json`. No model was re-run — the recomputation reuses the
  stored 2,080-pair curves in `results/allpairs_raw.npz`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 31 display / 663 inline equations / 30 figures; RESULTS 30 figures; 0 problems).
- **No `STOP`**: the direction is looping again after new feedback; zero unaddressed feedback files
  remain as of this entry, but the S24 plan candidates are still open.

## 2026-08-12 (later) — S24a: the readout was moved off the patched character, and the "decision basin" description narrowed

- **New experiment (the last untested control in Matthew's assay as run here).** Every previous number
  in both deliverables patched the final sequence position and read the final logits, fusing "the
  network's state switches sharply" with "the state at the position being read switches sharply".
  `experiments/pos_assay.py` separates them: the varied character stays at position 14 of
  `"The house was "` and $k$ filler characters (`" and then"[:k]`, $k\in\lbrace0,1,2,4,8\rbrace$) are
  appended, so the readout sits $k$ characters downstream of the patch. Injection moves to the residual
  stream *entering* block 0 — the only site that keeps both endpoints exact at a non-final position —
  and each checkpoint also gets an anchor row measured the report's standard way.
- **Model.** The reference character run was retrained from scratch (`train_frozen.py --tag ref_pos`,
  nothing frozen, seed 1337, 30,000 steps, 29.2 min) because `/tmp` had been wiped; step-0,
  matched-accuracy (step 2,500, val acc 0.5522) and step-30,000 checkpoints were all scored.
- **Four predictions, pre-registered in PLAN.md before the untrained and step-30,000 rows existed, all
  held.** (1) The logits at the patched position are identical for every $k$ (median $w$ = 0.2427
  throughout; worst endpoint reconstruction error 1.9e-5). (2) Trained median $w$ stays below 0.55 at
  every $k$: **0.243 / 0.290 / 0.249 / 0.244 / 0.257** for $k$ = 0/1/2/4/8, and paired against $k=0$ the
  offsets 2, 4, 8 are **not** distinguishable ($p$ = 0.27, 0.43, 0.22). (3) The untrained network gives
  the straight line at every $k$ (0.809/0.807/0.804/0.804/0.807, 0/150 plateaus), so trained-vs-init is
  $-0.51$ to $-0.57$ at $p=2.3\times10^{-26}$ everywhere. (4) Endpoint separation falls 44.5 → 16.4
  logit units and endpoint-decision disagreement collapses 86.7% → **8.7%** at $k=4$, where **52.0%**
  of pairs still meet the strict plateau rule.
- **Claim changed in both deliverables (old → new).** Summary/Headline and Conclusion previously read
  "plateaus … are next-character decision basins", with "decision basin" qualified only as a
  description rather than a mechanism. They now read: at the patched position the plateau looks like a
  next-character decision basin, but that is a description read off one token slot — four characters
  downstream the decision is gone for 91.3% of pairs while the sharp switch remains for 52.0% of them,
  so what switches is the network's state. No previous number is superseded; this narrows what the
  existing numbers are evidence *for*.
- **New supporting result.** The distance-independence is built late in training: at the
  matched-accuracy checkpoint the widths still degrade with offset (0.328/0.363/0.379/0.434/0.391,
  every $k>0$ wider than $k=0$, up to $+0.094$ at $p=5.6\times10^{-20}$; strict rate 28.0% → 7.3%),
  and by step 30,000 that penalty is gone.
- **Reproduction check (free by-product).** The freshly retrained reference gives anchor median widths
  **0.803 / 0.4428 / 0.3507** at init / matched / step 30,000, reproducing the reference run quoted
  throughout both deliverables (0.803 / 0.443 / 0.351) from a fresh run of the same recipe.
- **New figure.** **Figure 28** `plots/pos_offset.png` (median path per offset; width vs offset against
  the untrained baseline; endpoint separation, strict plateau rate and endpoint-decision disagreement
  vs offset). The exploratory Figures 28–30 are renumbered **29–31** in both files, with their prose
  citations updated; both files now hold 31 embeds, 31 visible captions and sequential numbering.
- **New Methods subsection (REPORT.md §Readout offset)** defining the offset $k$, the embedding-entry
  injection site and why `resid_post` cannot be used off the final position, the `read_final` /
  `read_patch` pair, endpoint separation $s=\lVert x_A-x_B\rVert_2$ and endpoint-decision disagreement.
- **Caveats updated, not added.** "Final-position interpolation only" was true when written and is not
  any more: Limitation 4 now records that the patch/readout separation has been tested to eight
  characters (one context, one filler, one seed) and Limitation 7 records that the decision does not
  even describe *where* the geometry appears. The `*(5)*` caveat in the all-pairs section was corrected
  the same way.
- **Code/data:** `experiments/pos_assay.py` (run to completion for the first time),
  `experiments/plot_pos.py` (middle-panel title corrected — at the final checkpoint the width does not
  widen with offset — and the right panel gained the endpoint-decision series); new
  `results/pos_assay.json`, `results/pos_assay_raw.npz`, `results/pos_assay.log`,
  `results/train_ref_pos.log`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 32 display / 733 inline equations / 31 figures; RESULTS 31 figures; 0 problems).
- **No `STOP`**: wall-clock remains and PLAN's S24 candidates (what the trainable blocks compute; a
  second model/tokenizer; a longer run) are still open. Zero unaddressed feedback files.

## 2026-08-12 (later) — S24b: chord linearization of individual MLP units — the bend is a few dozen units per path

**What changed in the deliverables.** A new Results section in both files, *"What the early MLPs
compute — a sparse, pair-dependent set of units carries the bend"* (RESULTS.md) / *"What the early
MLPs compute: a few dozen units per path, recruited from a shared pool"* (REPORT.md), plus a REPORT.md
Methods subsection defining the intervention and its two metrics. This is the first result here about
*what* the implicated blocks compute rather than where the computation may sit; PLAN S24 item 1 had
listed it as the direction's real open problem since 2026-08-03.

- **New intervention.** For a set $S$ of MLP hidden units in blocks 1–4, each unit's post-GeLU
  activation at the patched position is replaced along the path by the chord between its own endpoint
  values, $a_j(t)\to(1-t)a_j(0)+t\,a_j(1)$ — the unit keeps its endpoint behaviour and loses only its
  curvature in $t$. Both endpoints stay exact for any $S$ (worst $|d(0)|$, $|1-d(1)|$ = **1e-6** over
  every pair and condition). Units ranked by
  $I_j=\lVert W^{\mathrm{proj}}_{:,j}\rVert_2\max_t|a_j(t)-\bar a_j(t)|$; three selection rules
  (per-pair top-$k$, one global top-$k$, random $k$) at 13 sizes, 150 pairs, block-0 interpolation,
  step-30,000 reference checkpoint — the same subsample and setting as the gain and per-block
  interventions, so widths are comparable (unmodified 0.351, untrained 0.803).
- **New numbers.** All 3,840 units linearized → median $w$ **0.743**, recovering **86.7%** of the
  trained→untrained gap (MLP deletion reaches 0.796). Per-pair top-32 (0.83% of units) → **50.9%**;
  top-512 → 83.6%; per-pair median $k$ for half of its own gap = **64** (IQR 32–128). Random 32 →
  **1.2%**; random selection needs ~2,048 units for what 32 ranked units give (≈64× concentration).
  One global set of 32 → **19.0%**; 668 of 3,840 units ever enter a top-32, 82% of those serve ≥2
  pairs, most reused unit 88/150 pairs, median overlap of a pair's top-32 with the global top-32
  **9/32**. Block composition of the 4,800 top-32 slots: 16.0 / 18.8 / 27.8 / 37.4% in blocks 1/2/3/4
  — increasing with depth, the opposite ordering to single-block MLP deletion (41/28/18/11%), which is
  explained in the section rather than left as a discrepancy.
- **Claims updated (no result superseded; nothing removed).** The hypothesis paragraph's
  "a **relocatable** computation that we have not yet characterised" (RESULTS.md) becomes "carried by a
  few dozen MLP units per path"; REPORT.md's Conclusion sentence "what the trainable blocks actually
  compute to bend the path is still uncharacterised" is replaced by the counts above, with "what those
  units detect" named as what stays open; Limitation 7 and the Interpretation paragraph are narrowed
  the same way; the Summary and Headline each gain the result as their closing paragraph.
- **New figure.** **Figure 29** `plots/neuron_path.png` — A: median width vs number of linearized units
  for the three selection rules; B: per-pair smallest $k$ for half that pair's gap; C: unit reuse across
  pairs. The exploratory Figures 29–31 are renumbered **30–32** in both files with their prose citations
  updated; both files now hold 32 embeds, 32 visible captions, sequential numbering 1–32.
- **Code/data:** new `experiments/neuron_path.py`, `experiments/plot_neuron_path.py`;
  new `results/neuron_path_summary.json`, `results/neuron_path_raw.npz`, `results/neuron_path.log`.
  No training was needed — the run took 198 s on the existing `ref_pos` checkpoints.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 35 display / 780 inline equations / 32 figures; RESULTS 32 figures; 0 problems).
- **No `STOP`**: wall-clock remains and PLAN S24 item 3 (a longer character run / second model) is open.
  Zero unaddressed feedback files.

## 2026-08-12 (S24c) — what the path-bending units detect: character tuning measured in ordinary text

- **Feedback check first (CLAUDE.md Part C).** Listed the direction root for `human_feedback*.md` /
  `*REVIEW*` without the `.addressed.md` suffix: none of the seven. This iteration advanced the plan
  (PLAN "Next step" named this experiment).
- **What was added.** `experiments/neuron_feature.py` characterises the units that Figure 29 found,
  using a data source outside the interpolation assay: the model's own 90% training split, tiled into
  7,842 non-overlapping 128-character windows (941,040 scored positions, the first 8 positions of each
  window dropped). Per block-1–4 hidden unit it accumulates the mean post-GeLU activation conditioned
  on the current character, standardizes that 65-character profile within the unit
  ($z_{c,j}$), and asks whether tuning at a pair's two endpoint characters predicts which units that
  pair recruited into its top-32. Forward passes only, 9 s, no training.
- **New numbers.** Differential tuning $|z_a-z_b|$ ranks recruited units at mean **AUROC 0.847**
  (99% CI 0.834–0.858, median 0.857), precision@32 **21.6%** = **26×** the 0.83% chance rate;
  $\max(z_a,z_b)$ 0.840 / 24.1%; assay-derived global importance (in-domain reference) 0.913 / 27.5%;
  overall-activity control **0.562** / 3.3%; random 0.498 / 0.79% (paired Wilcoxon vs random
  $p=2.3\times10^{-26}$; vs the activity control $p=2.3\times10^{-26}$). Recruitment falls
  monotonically **4.9% → 0.09%** across deciles of differential tuning. A recruited unit's preferred
  character is one of the pair's endpoints for **27.2%** of recruitments vs a **2.8%** base rate
  (**9.8×**, $p=2.3\times10^{-26}$). The 668 pool units have median sharpness $\max_c|z_c|$ **5.45**
  vs **4.47** for the other 3,172 (Mann–Whitney $p=5.8\times10^{-27}$). Robustness: re-standardizing
  over the 62 characters with ≥100 corpus occurrences and keeping the 143 pairs built from them gives
  **0.858** / 21.4%, so rare-character noise is not driving it. Qualitative: the most reused unit
  (block 2, 88/150 pairs) is a capital-letter detector whose top contexts are proper-name onsets
  (`DUCHESS OF Y`, `Duke of Y`, `Henry the F`).
- **Claims updated (no result superseded; nothing removed).** The four places that named "what those
  units detect" as the open part — REPORT.md Summary, Conclusion, Limitation 7, and the RESULTS.md
  hypothesis paragraph and Headline — now carry the tuning result, with the correlational nature of
  the link stated in each. Old wording "…a few dozen gated units per path … what the individual units
  detect remains open" → new "…identifiable in advance as the character detectors tuned to the pair's
  own endpoints (AUROC 0.847), though that link is correlational".
- **New figure.** **Figure 30** `plots/neuron_feature.png` — A: AUROC per ranking rule (per-pair
  points); B: recruitment rate by tuning decile; C: tuning sharpness, pool vs never-recruited;
  D: tuning profiles of the three most-reused units. The exploratory Figures 30–32 are renumbered
  **31–33** in both files with their prose citations updated; both files now hold 33 embeds, 33 visible
  captions, sequential numbering 1–33.
- **Code/data:** new `experiments/neuron_feature.py`, `experiments/plot_neuron_feature.py`;
  new `results/neuron_feature_summary.json`, `results/neuron_feature_raw.npz`,
  `results/neuron_feature.log`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 39 display / 808 inline equations / 33 figures; RESULTS 33 figures; 0 problems). Caption
  check: 33 embeds / 33 `**Figure` captions in each file.
- **No `STOP`**: wall-clock remains and PLAN S24 item 3 (a longer character run / second model) is
  open. Zero unaddressed feedback files.

## 2026-08-12 (S24c, second half) — the corpus tuning rule made causal and held out

- **What was added.** `experiments/neuron_feature_causal.py` feeds the corpus-tuning ranking back into
  the chord-linearization intervention: for each pair, the $k$ units with the largest differential
  tuning $|z_a-z_b|$ are linearized, selected with no quantity from the assay (not $d(t)$, not $I_j$,
  not the pair's own curve). 22 s of forward passes, no training.
- **New numbers.** Corpus-selected top-32 removes **28.9%** of the trained→untrained width gap (median
  $w$ 0.351 → **0.482**, 98% of pairs widen) against **1.2%** for 32 random units
  ($p=2.5\times10^{-26}$) and **19.0%** for the assay-derived global set, the strongest previous
  pair-blind rule ($p=2.7\times10^{-11}$); it stays below the fitted per-pair ceiling of 50.9%
  ($p=7.3\times10^{-26}$). At $k$ = 8 / 128 / 512 the held-out rule gives 11.0 / 47.5 / **66.4%**
  against the global set's 8.6 / 30.9 / 57.4% and random's 0.1 / 3.1 / 15.8%. Built-in checks: the
  unmodified baseline reproduces per pair exactly (0.3507, max per-pair difference 0.000000) and the
  worst endpoint deviation is $10^{-6}$.
- **Claims updated (no result superseded).** The caveat added earlier this iteration — "correlation
  between two measurements, not an intervention" — is now **wrong** and was replaced in both files by
  the residual it leaves ("the corpus-selected edit removes 28.9% where the fitted ranking's removes
  50.9%, so tuning names much of the responsible population and not all of it"). REPORT Summary,
  Conclusion and Limitation 7 and the RESULTS.md hypothesis paragraph and Headline gained the causal
  number.
- **New figure.** **Figure 31** `plots/neuron_feature_causal.png` — recovered fraction vs $k$ for the
  held-out corpus rule against the fitted per-pair, global and random rules. Exploratory Figures 31–33
  renumbered **32–34**; both files now hold 34 embeds, 34 visible captions, sequential numbering 1–34.
- **Code/data:** new `experiments/neuron_feature_causal.py`,
  `experiments/plot_neuron_feature_causal.py`; new `results/neuron_feature_causal.json`,
  `results/neuron_feature_causal_raw.npz`, `results/neuron_feature_causal.log`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 39 display / 832 inline equations / 34 figures; RESULTS 34 figures; 0 problems).

## 2026-08-12 (S24d) — the residual half of the path-bending units, and a refuted fix

- **What was added.** `experiments/neuron_bigram.py` — one further pass over the same 90% training
  split, tabulating each block-1–4 hidden unit's mean post-GeLU activation against the (previous,
  current) character pair. It answers the residual the previous iteration left open (corpus-selected
  32 units remove 28.9% of the width gap; the fitted per-pair 32 remove 50.9%) with a description of
  what the missed units are, and a causal test of the obvious fix. 19 s of forward passes, no training.
- **New numbers.** (i) *Descriptive:* for recruited units the character ranking finds (top decile of
  $D_j$), the current character explains a median **96%** of the corpus response; for the ones it
  misses, **51%**, with the interaction share rising 18% → **49%** (Mann–Whitney
  $p=1.4\times10^{-186}$); the population median is 37%. (ii) *Matched-size ablation:* 8 missed
  recruits remove **11.5%** of the gap against **29.1%** for 8 found recruits (paired
  $p=1.2\times10^{-20}$, 138 pairs). (iii) *The fix fails:* the context-matched (previous character =
  space) profile ranks the whole population better — mean AUROC **0.886** vs **0.869** for the
  current-character rule on the same 84 pairs ($p=1.4\times10^{-5}$) — and selects worse: its top 32
  remove **21.9%** of the gap against **31.9%** (paired $p=1.9\times10^{-11}$), with the fitted
  ceiling at 52.6% and random at 0.6% on those pairs; precision@32 foretells it (20.3% vs 25.6%).
  Built-in checks: marginalizing the bigram table reproduces the previous section's tuning scores $z$
  to **0.0000**, the unmodified baseline reproduces per pair to 0.3507 (max difference 0.000000), and
  the worst endpoint deviation is $10^{-6}$.
- **Claims updated (no result superseded).** The caveat "what the remaining half responds to is not
  established here" (REPORT Results + Caveats, RESULTS Caveats) was **true and is now answered**; both
  files point to the new subsection instead. REPORT Summary, Conclusion and Limitation 7, and the
  RESULTS.md hypothesis paragraph, gained the context-dependence result and the negative on bigram
  conditioning. The older caveat "a unit that responds to a longer pattern would be summarized
  crudely" now points forward to the measurement of how many such units there are.
- **New figure.** **Figure 32** `plots/neuron_bigram.png` — (a) cumulative current-character variance
  share for found / missed / all units, (b) matched-size ablation of found vs missed recruits, (c)
  five selection rules at $k=32$ re-scored on the 84 like-for-like pairs. Exploratory Figures 32–34
  renumbered **33–35**; both files now hold 35 embeds, 35 visible captions, sequential numbering 1–35.
- **Code/data:** new `experiments/neuron_bigram.py`, `experiments/plot_neuron_bigram.py`; new
  `results/neuron_bigram_summary.json`, `results/neuron_bigram_raw.npz`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 43 display / 873 inline equations / 35 figures; RESULTS 35 figures; 0 problems).

## 2026-08-12 (S24e) — a fitted probe reads the missed units, and the real limit turns out to be the standardization

- **What was added.** `experiments/neuron_probe.py` (21 s of forward passes, no training) fits, for
  every block-1–4 hidden unit, a ridge regression predicting its post-GeLU activation from the eight
  characters ending at the position plus a full previous×current interaction table, on the model's own
  90% training split, with windows split 80/10/10 so the penalty is chosen on one held-out slice and
  every $R^2$ is reported on another. It replaces S24d's hand-built conditioning with a fitted one and
  turns it into a new blind selection rule evaluated at the assay's own context.
  `experiments/neuron_probe_control.py` (23 s) adds the two controls that decompose that rule.
- **New numbers.** (i) *Describability:* the recruits the character rule misses reach median held-out
  $R^2$ **0.29 → 0.53 → 0.78** across window lengths 1 → 8 → 8+interaction, against **0.92 → 0.93 →
  0.97** for the recruits it finds (Mann–Whitney $p=1.5\times10^{-116}$ on the full model,
  $p=8.3\times10^{-185}$ for the context gain) — so "context-dependent" is literal and local, not
  diffuse. (ii) *New blind rule:* scoring units by the probe's predicted activation difference at
  `"The house was ␣X"` and linearizing the top 32 removes **56.5%** of the trained→untrained width gap,
  against 28.9% (standardized character rule), 22.5% (bigram rule), 19.0% (global assay-derived set)
  and 1.2% (random), all paired $p\le2.3\times10^{-26}$; it also clears the pair-fitted top-32's 50.9%
  (paired $p=2.3\times10^{-17}$), so that ranking was never a ceiling. (iii) *Decomposition:* the
  character profile with per-unit standardization removed reaches **56.3%** on its own, while the
  probe's prediction re-standardized reaches **34.8%** — the scale is worth ~27 points and the fitted
  context ~6, together 0.2 more than scale alone (paired $p=0.0022$); median overlap between the probe
  rule's and the character rule's top-32 is 9 units. Built-in checks: the unmodified baseline
  reproduces per pair to 0.3507 (max difference 0.000000) and the worst endpoint deviation is
  $10^{-6}$.
- **Claims updated (no result superseded).** No earlier number changed; S24d's *interpretation* is
  corrected in place. Where both deliverables said a 32-unit corpus rule is limited by "precision at
  the top of the ranking", they now also say the larger limit was the per-unit standardization both
  earlier rules applied, and that the fitted per-pair ranking is not a ceiling at $k=32$. The caveat
  "a richer conditioning — more history, or a learned feature — might yet select them well" was
  **true and is now tested**; both files carry the outcome. REPORT Summary, Conclusion and Limitation
  7, and the RESULTS.md hypothesis paragraph and Headline, gained the new result.
- **New figure.** **Figure 33** `plots/neuron_probe.png` — (a) median held-out $R^2$ vs probe window
  length for found / missed / all units, (b) seven blind selection rules at $k=32$ against the
  pair-fitted reference, (c) recovered fraction vs $k$ for five rules. Exploratory Figures 33–35
  renumbered **34–36**; both files now hold 36 embeds, 36 visible captions, sequential numbering 1–36.
- **Code/data:** new `experiments/neuron_probe.py`, `experiments/neuron_probe_control.py`,
  `experiments/plot_neuron_probe.py`; new `results/neuron_probe_summary.json`,
  `results/neuron_probe_raw.npz`, `results/neuron_probe_control.json`,
  `results/neuron_probe_control_raw.npz` and their logs.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 46 display / 928 inline equations / 36 figures; RESULTS 36 figures; 0 problems).

## 2026-08-12 — S24f: the text-only selection score is at its ceiling (write norm and oracle both fail to beat it)

- **Context.** Zero unaddressed `human_feedback*`/`*REVIEW*` files (all seven end in `.addressed.md`),
  so this iteration advanced the plan; PLAN's own "Next step" named this consolidation — re-rank the
  report's selection rules by *displacement* rather than preference. One script, no training, 62 s of
  forward passes on the same reference character run (step 30,000, seed 1337) and the same 150 pairs.
- **New experiment (`experiments/neuron_scale.py`), pre-registered before running.** Five more blind
  selection rules through the identical chord intervention at $k\in\{8,32,128\}$: the character
  profile and the fitted probe each multiplied by the unit's write norm
  $n_j=\lVert W_{\mathrm{proj}}[:,j]\rVert_2$; the *measured* endpoint swing
  $E_j=|a_j(1)-a_j(0)|$ (an oracle for what the corpus rules estimate) with and without $n_j$; and
  $n_j$ alone as a pair-blind floor. Predictions registered: (P1) the write norm adds $\ge 2$ points;
  (P2) the oracle beats every corpus rule; (P3) $n_j$ alone recovers $<20\%$.
- **Outcome — P1 and P2 refuted, P3 held more strongly than predicted.** At $k=32$, recovered fraction
  of the trained→untrained width gap: write-norm weighting takes the character rule **56.3% → 55.4%**
  (paired $p=0.049$), the probe 56.5% → 56.6% in median but worse on 62% of pairs individually (mean
  width change $-0.003$, paired $p=2.7\times10^{-4}$), and the oracle **56.6% → 55.3%** (paired
  $p=1.1\times10^{-9}$); write norms span only **1.71×** between the 5th and 95th percentiles (median
  1.66, IQR 1.49–1.82). The oracle **ties** the text-only probe (56.6% vs 56.5%, paired $p=0.27$), so
  the corpus rules are not estimation-limited at $k=32$. Weighted endpoint displacement beats the
  pair-fitted curvature ranking (**55.3%** vs **50.9%**, paired $p=2.2\times10^{-17}$) while sharing a
  median 20 of 32 picks. The write norm alone removes **0.3%** at $k=32$ (below random's 1.2%) and
  12.0% at $k=128$. Built-in checks: baseline reproduces the stored per-pair widths exactly (median
  0.3507, max difference 0.000000), worst endpoint deviation $10^{-6}$.
- **Claims updated (no result superseded).** No earlier number changed. Both deliverables previously
  ended the selection thread at "keep each unit's own activation scale"; they now add that the score is
  at the ceiling of its family, name the remaining limit as the score's *form* (individual rather than
  joint displacement) rather than estimate quality, and state the negative write-norm result. REPORT
  Summary, Conclusion and Limitation 7, and the RESULTS.md hypothesis paragraph and Headline, carry it.
- **New figure.** **Figure 34** `plots/neuron_scale.png` — (a) ten selection rules at $k=32$,
  (b) empirical CDF of the per-pair width change caused by write-norm weighting for three scores,
  (c) recovered fraction vs $k$ for the oracle, the character rule, the pair-fitted ranking, the
  pair-blind write norm and random. Exploratory Figures 34–36 renumbered **35–37**; both files now hold
  37 embeds, 37 visible captions, sequential numbering 1–37.
- **Code/data:** new `experiments/neuron_scale.py`, `experiments/plot_neuron_scale.py`; new
  `results/neuron_scale_summary.json`, `results/neuron_scale_raw.npz`, `results/neuron_scale.log`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 48 display / 981 inline equations / 37 figures; RESULTS 37 figures; 0 problems).

## 2026-08-12 — S24g: unit interactions measured — worth 3.4 points at $k=128$, nothing at $k=32$

- **Feedback status.** Seven `human_feedback*` files in the direction root, all already
  `.addressed.md`; zero unaddressed, so this iteration advanced the plan (PLAN's own "Next step" named
  this experiment).
- **New experiment (`experiments/neuron_greedy.py`), pre-registered before running.** Residual-corrected
  (greedy) selection: build the linearized set in $R$ equal rounds and, before each round, re-measure
  every unit's importance $I^{S_r}_j=n_j\max_t|a^{S_r}_j(t)-\mathrm{chord}_j(t)|$ *with the units
  already chosen linearized*, so a unit is scored by the bend still left. $R\in\{1,2,4,8\}$,
  $k\in\{32,128\}$, same 150 pairs / context / checkpoint. $R=1$ is exactly the one-shot pair-fitted
  rule, so it is both the control and a free reproduction check. Predictions registered: (P1) $R=4$ at
  $k=32$ gains $\ge 5$ points over $R=1$; (P2) recovery is non-decreasing in $R$ at both $k$; (P3)
  $R=8$ at $k=32$ clears the 56.6% of the best per-unit rule.
- **Outcome — P1 and P3 refuted, P2 holds only at $k=128$.** Recovered fraction of the
  trained→untrained width gap. At $k=32$: **50.9% ($R=1$) → 51.3% → 49.8% → 49.8% ($R=8$)**, paired
  $p=0.24$, $0.41$, $0.43$ against $R=1$; only 50.7% of pairs not worse. At $k=128$: **68.4% → 70.7% →
  71.1% → 71.8%**, paired $p=9.4\times10^{-17}$, $1.5\times10^{-19}$, $6.1\times10^{-21}$, with
  $R=4\rightarrow8$ itself significant ($p=6.0\times10^{-6}$) and 84.7% of pairs not worse (median
  width gain $+0.0145$). Set overlap with the one-shot picks at $R=8$: median **100/128** and
  **26/32**. Built-in checks, all exact: $R=1$ reproduces `neuron_path.py`'s per-pair pair-fitted
  widths (max difference 0.000000 at both $k$), baseline reproduces to median 0.3507 (max difference
  0.000000), worst endpoint deviation $10^{-6}$.
- **Claims updated (no result superseded).** No earlier number changed. Both deliverables previously
  left the residual gap attributed to an unmeasured *joint* effect; they now bound it — joint selection
  closes **3.4 of the 18.3 points** (≈ a fifth) between the best one-shot ranking and the 86.7%
  all-units ceiling at $k=128$, and none at $k=32$. This also corrects the S24f reading: the blind
  text-only rule beats the fitted ranking at $k=32$ because of the per-unit score's *form* (endpoint
  displacement vs path curvature), not because the fitted ranking misses joint structure — a rule that
  sees joint structure perfectly does no better there. REPORT Summary, Conclusion and Limitation 7, and
  the RESULTS.md Headline, carry it.
- **New figure.** **Figure 35** `plots/neuron_greedy.png` — (a) recovered fraction vs rounds $R$ for
  $k=32$ and $k=128$, against the best per-unit rule at each size and the all-units ceiling;
  (b) empirical CDF of the per-pair width change from $R=8$ vs $R=1$; (c) median overlap of the
  sequential set with the one-shot set vs $R$. Exploratory Figures 35–37 renumbered **36–38**; both
  files now hold 38 embeds, 38 visible captions, sequential numbering 1–38.
- **Code/data:** new `experiments/neuron_greedy.py`, `experiments/plot_neuron_greedy.py`; new
  `results/neuron_greedy_summary.json`, `results/neuron_greedy_raw.npz`, `results/neuron_greedy.log`.
- **Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → **ALL CHECKS PASS**
  (REPORT 49 display / 1,057 inline equations / 38 figures; RESULTS 38 figures; 0 problems).
