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

## 2026-07-29 — operator feedback #1: extended k to a true step function; renamed metrics; restructured tables

**Feedback addressed** (`human_feedback_1.txt` -> `human_feedback_1.addressed.md`). Two points, both
acted on:

1. *"the k you used were too small — 3 of them are near linear, use maybe another 5 bigger k, the
   biggest one should look like a step function. Do the experiments and rewrite the report."*
2. *"The tables are really confusing. Duplicated column names. Also I saw R^2 which is the same as the
   famous linear R^2 — use a better notation."*

**What changed in RESULTS.md / REPORT.md.**

- **Experiment extended from 5 to 10 sharpness settings.** Added $k \in \{20, 40, 80, 160, 320\}$ to the
  existing $\{0.5, 1, 2, 5, 10\}$, trained with identical inputs/hyperparameters, 3 seeds each — 15 new
  models on the primary 1000-image grid plus 15 more on the 10,000-image robustness grid, and the full
  201-point brightness sweep on both checkpoints for all of them. At $k=320$ the target's 10–90%
  transition width is $0.0046$ against a probe-grid spacing of $0.003$, so it is a step function at the
  resolution at which we measure, and its concentration score sits at the theoretical maximum.
- **The conclusion is unchanged but the evidence is much stronger, and the reason has changed.** Before,
  the deepest-layer concentration was still rising at the largest $k$ and the report had to argue from a
  gap to the target. Now the curve visibly *saturates*: deepest-layer $\Gamma_3$ = 1.455 ± 0.036 ($k$=10)
  -> 1.491 ± 0.068 ($k$=20) -> 1.483 -> 1.468 -> 1.451 -> **1.458 ± 0.189** ($k$=320), i.e. flat across a
  16x further sharpening, while the target reference goes 2.70 -> **5.00** (its ceiling). Flank share
  $\Phi_3$ likewise bottoms out at 0.265 ($k$=10) and *rises* back to 0.283 ($k$=320) against a target
  $\Phi$ of exactly 0.
- **New decisive control, now the report's strongest single result.** On the 10,000-image grid at
  $k=320$ the model *output* becomes a genuine switch — $\Gamma_{\text{out}} = 4.13$ of a maximum 5.00,
  $\Phi_{\text{out}} = 0.005$, sweep $R^2 = 0.848$ — while the deepest hidden layer stays at
  $\Gamma_3 = 1.659 \pm 0.168$, $\Phi_3 = 0.279$. Output 78% of the way to a perfect plateau,
  representation 16%. This removes the "the model never learned the step" confound that the previous
  version could only partly rule out.
- **Non-monotonicity reported as observed:** on the 10k grid $\Gamma_3$ peaks at 2.024 ($k$=40) and falls
  back to ~1.66 for the three step-like settings. Not smoothed over (PLAN.md: "do not force a monotonic
  interpretation").
- **Metric renamed to fix the $R^2$ collision (feedback point 2).** Concentration score
  $R_l(k) \to \Gamma_l(k)$ ("concentration gain") and flank movement fraction $F_l(k) \to \Phi_l(k)$
  ("flank share"). $R^2$ now appears in the deliverables *only* as sweep goodness-of-fit. Previously the
  table header "$R_2$" (layer-2 concentration) was visually indistinguishable from $R^2$.
- **Tables restructured to remove duplicated/ambiguous column names (feedback point 2).** The old single
  wide table mixed "target $R$ / prediction $R$ / $R_1$ / $R_2$ / $R_3$ / target $F$ / $F_3$ / sweep
  $R^2$" in one row. It is now four tables, each with one metric and fully-spelled column names: Table 1
  = concentration gain $\Gamma$ (target curve | model output | hidden layer 1 | hidden layer 2 | hidden
  layer 3), Table 2 = flank share $\Phi$ (same columns), Table 3 = fit and training diagnostics, Table 4
  = the 10,000-image grid. Every column carries its own metric symbol; no name repeats within a table.
- **Both $\Gamma$ scale endpoints now stated everywhere**: $\Gamma = 1$ uniform, $\Gamma = 5$ maximum, so
  results are quotable as "% of the way from uniform to a perfect plateau". This ceiling was implicit
  before and is what makes the saturation legible.
- **Adequacy caveat added.** All 30 runs pass the gate ($\rho_{\text{val}}$ 1.017–1.109, training loss
  within 0.1% of its minimum), but at $k=160,320$ the validation minimum falls at epoch 15 of 2000, so
  the min-validation checkpoint there is a barely-trained network. Stated in RESULTS.md Table 3 notes and
  REPORT.md §2 and §6 rather than left for a reader to find.
- **REPORT.md rewritten end to end** around the extended result: Summary now leads with the saturation
  and the output-vs-representation gap; Methods gains a "why $k$ runs to 320" subsection with transition
  widths, the $\Gamma_{\max}=5$ / $\Phi_{\min}=0$ baseline, and the model-output reference promoted to a
  named baseline with its own equation; Results reorganised into 7 numbered sections.

**Figures.** All regenerated for 10 settings. Because $k$ is an *ordered* variable with 10 levels, the
5-hue categorical palette no longer applies (CLAUDE.md rule 13): $k$-indexed series now use the
CVD-designed sequential `cividis` map (dark blue -> yellow, monotone in lightness, readable in grayscale)
with a cycling line style so colour is never the sole identity channel. Layer-indexed series keep the
5-hue CVD palette. `target_functions.png` gained a transition-zoom panel; `prediction_sweeps.png` is now
2x5; `training_curves.png` validation panel is log-scale. **New:** `saturation_and_fit.png` (Figure 6),
which carries the decisive control — concentration gain at both training-set sizes beside sweep $R^2$.
`checkpoint_robustness.png` narrowed to the checkpoint comparison alone, its old training-size panel
being superseded by Figure 6. All 8 figures embedded with visible numbered captions in both deliverables;
`check_render.py` passes (12 display eqs, 8 figures each, 0 problems).

**Superseded numbers** (old -> new, deepest layer, primary grid): the reported endpoint of the sweep was
$R_3(k{=}10) = 1.455 \pm 0.036$ against a target of 2.70; it is now $\Gamma_3(k{=}320) = 1.458 \pm 0.189$
against a target of 5.00. On the 10,000-image grid, $R_3(k{=}10) = 1.823 \pm 0.222$ (output 2.42) ->
$\Gamma_3(k{=}320) = 1.659 \pm 0.168$ (output 4.13). The $k \le 10$ rows themselves are unchanged — the
models and analysis for those settings were not re-run.
