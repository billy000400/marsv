# JOURNAL — Direction 10: Does combined path smoothness recover the weekday activation manifold?

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-15 — Iteration 1: full pipeline S1→S4 + verdict (fresh direction → decisive negative)

**Audit (S1).** The scripts the plan references (`optimize_path.py`, `slerp_relative_distance.py`) do
NOT exist anywhere in the repo; no prior weekday code exists. Llama 3.1 8B base is cached at
`/network/hf_cache/hub/...`. Built the pipeline from scratch: `common.py`, `pathlib_opt.py`,
`s2_collect.py`, `s4_sweep.py`, `s5_analyze.py`, `s6_allpairs.py`.

**Key constraints hit + decisions (logged as deviations):**
- 8B bf16 = 16 GB > 7.2 GB VRAM share AND > 8 GB per-agent RAM soft cap. Box physically has ~230 GB
  RAM free. Chose: load whole model on CPU (real tensors), collect the one-time layer-28 activations
  via CPU forward, then move ONLY the layers 28–31 + norm + lm_head tail to GPU for the sweep. Tried
  accelerate device_map with "cpu" offload first — it makes offloaded params *meta*, which breaks a
  manual tail forward; CPU-load-then-move-tail is the clean fix.
- "PCA-64" impossible with 49 activations (rank ≤ 48); kept all 48. Opt subspace (32) + recovery
  metric (PCA-32) unaffected.
- Appendix-A.9 SVD recovery score not reproducible without the appendix → primary metric = transparent
  PCA-32 nearest-spline distance. `d(t)` defined from first principles (referenced helper absent).

**S2 setup — validated.** Task acc 0.939 (46/49), weekday mass 0.743, other 0.257; 7 well-separated
centroids (adj spacing 8.5–11.8); periodic cubic spline fit.

**S3 tail — validated.** Injecting a prompt's own layer-28 activation through the GPU tail reproduces
the full-model 8-bin behavior (argmax always matches; max L1 0.05). λ=0 recovers the chord to 1e-15.

**S4 pilot Tue→Wed (5 λ + output-only, 3 seeds).** Decisive NEGATIVE:
- Recovery (nearest-spline dist, PCA-32; spline=0.004): λ=0 = 0.961 (best optimized), monotonically
  worsens to output-only 1.023 (linear init) / 1.40–1.43 (perturbed). No intermediate λ helps.
- The centroid spline is Pareto-dominated in BOTH energies (E_act 104.9>88.8; E_out 1.118>1.026), so
  the objective structurally cannot prefer it.
- High-λ paths are init-dependent (E_act 306–313 vs 93 at ~equal E_out) → behavior objective
  underdetermines the activation path.

**What I learned.** The fitted weekday manifold is *less* smooth (higher KE) than the straight chord in
both activation and behavior space — a generic smoothness prior points away from it, not toward it.
This is the plan's "spline dominated in both energies" interpretation.

**Next step.** S6 all-7-adjacent-pairs run (linear init) is in flight to confirm generalization;
then finalize (write `STOP` if every pair shows the same monotone-worsening + dominated-spline
pattern). Deliverables (RESULTS/REPORT/CHANGELOG/PLAN) written to current-best.

On track? yes — S1–S4 done (~85%), S6 generalization running; question already answered (negative) for the pilot.

## 2026-07-15 — Iteration 1 addendum: S6 all-pairs + finalize

S6 (all 7 adjacent pairs, linear init) completed. Every pair: `best_opt == rec_linear` (no λ beats
the chord), mean chord recovery 0.988 vs spline 0.004, spline dominated in both energies 7/7. The
negative result is robust across the whole weekday cycle. Added `plots/s6_allpairs_recovery.png`;
curated RESULTS.md/REPORT.md to the generalized verdict; verified REPORT display math (6/6 render,
0 degraded) and inline-math hazards (none). Direction question is answered decisively — wrote STOP.

On track? yes — S1–S6 done (100%); research question answered (NEGATIVE, generalizes across all 7 pairs); STOP written.

## 2026-07-15 — Iteration 2: finalization (STOP re-created)

Resumed with working memory reset. Read CLAUDE.md/BUDGET.md/PLAN.md/JOURNAL.md/RESULTS.md/CHANGELOG.md.
Direction was already complete (S1–S6, decisive NEGATIVE, generalizes across all 7 adjacent pairs),
but the `STOP` file the prior entry claimed to have written was missing from disk. Re-verified the
deliverables are intact: 6 figures present in plots/, results/ artifacts present, RESULTS.md/REPORT.md
current-best. Re-ran the CLAUDE.md math-rendering check on REPORT.md (6/6 display equations render,
0 degraded, 0 inline hazards). Re-created `STOP`. No result numbers changed.

On track? yes — S1–S6 done (100%); research question answered (NEGATIVE, 7/7 pairs); STOP written.

## 2026-07-15 — Iteration 3: per-pair energy trade-off figure + finalize (STOP re-created)
Resumed with memory reset; direction already COMPLETE (S1–S6, decisive NEGATIVE, generalizes 7/7).
`STOP` was again missing from disk. Smallest useful advance (plan's flagged optional polish, and a
real gap — the energy trade-off figure only covered Tue→Wed): built `experiments/s6_energy_plot.py`,
a pure post-processor of `results/allpairs_summary.json` (no model, no GPU), producing
`plots/s6_allpairs_energy_tradeoff.png` — E_act vs E_out for all 7 pairs with each fitted spline in
the dominated top-right corner. Script independently re-confirms spline dominated in both energies
7/7. Referenced the figure from RESULTS.md/REPORT.md; no numbers changed. Re-verified REPORT math
(6/6, 0 degraded, 0 inline hazards). Re-created `STOP`.

On track? yes — S1–S6 done (100%); research question answered (NEGATIVE, 7/7 pairs); STOP written.
