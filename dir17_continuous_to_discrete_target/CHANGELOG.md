# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-27 — first full experiment; RESULTS.md + REPORT.md written from the template

**What changed.** Both deliverables went from empty templates ("TODO — describe this direction") to the
complete experiment. Nothing was superseded — these are the first numbers this direction has produced.

**New in RESULTS.md / REPORT.md.**
- Full 5-$k$ × 3-seed brightness-regression experiment (S1–S4 of PLAN.md), primary numbers from the
  final checkpoint over 100 digit-balanced held-out MNIST test images.
- Headline: deepest-layer concentration ratio $R_3$ rises monotonically 1.094 ± 0.010 ($k$=0.5) →
  1.455 ± 0.036 ($k$=10) while the target's own ratio rises 1.01 → 2.70; flank movement fraction $F_3$
  falls only 0.356 → 0.265 against the target's 0.397 → 0.048. Verdict: a switch-like *continuous*
  target is **not sufficient** to produce classification-style activation plateaus.
- Metric $F_l(k)$ (flank movement fraction, outer 40% of the brightness range) added beyond PLAN.md's
  required metrics, because $R_l$ alone cannot distinguish "somewhat more concentrated" from "actually
  flat"; it carries the negative half of the verdict.
- Robustness: min-validation-loss checkpoint reproduces the final checkpoint ($R_3(k{=}10)$ 1.455 vs
  1.451). Secondary 10,000-image grid (fails the adequacy gate — no overfitting, $\rho_{val}\approx1.005$)
  fits far better (sweep $R^2$ 0.98 vs 0.89) and shows a **larger** effect, $R_3(k{=}10) = 1.823 \pm 0.222$,
  $F_3 = 0.204$ — so the primary numbers are a lower bound on effect size, and the verdict is unchanged.

**New figures (all embedded in both deliverables):** `target_functions.png`, `training_curves.png`,
`prediction_sweeps.png`, `activation_movement_by_k.png`, `concentration_vs_k.png`,
`checkpoint_robustness.png`, `main_summary.png`.

**Deviation from PLAN.md worth recording.** PLAN.md recommended "all training images" but also requires
slight validation overfitting as an adequacy gate; the two conflict (a 1-D brightness target on 50k
images does not overfit). Primary runs therefore use 1000 digit-balanced training images — this
project's canonical MNIST plateau setting, and the regime that passes the gate — with the 10,000-image
grid reported as the secondary check described above. Also added a cosine LR decay to the global
schedule (applied identically to all $k$), without which training loss ended mid-oscillation and failed
the smooth-convergence condition ($\rho_{train}$ up to 8995 → 1.00–1.13 with the decay).
