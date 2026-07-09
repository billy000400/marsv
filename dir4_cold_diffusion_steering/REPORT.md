# ColdSteer — on-manifold correction for activation steering

> Final, presentable, current-best only (history in CHANGELOG.md). This report is split into four
> short, topic-focused parts so each can be read on its own. This file is the index: it states the
> overall question, the headline numbers, and points to each part. Read the parts in order for the
> full story, or jump to the one you care about.

## Summary — the one-paragraph version

**Activation steering** controls a language model's behavior at inference time: find a direction
`v` in the hidden state that encodes a concept (e.g. "positive sentiment"), then add `α·v` to the
activations as the model runs, where `α` sets the strength. Pushing hard on `α` drags the
activation **off-manifold** — away from the activations the model actually produces on real text —
and fluency collapses. This direction asks whether a small **corrector** can preserve the steering
effect while keeping the activation safe for the model. The central finding: the corrector that
works keeps the steering projection exactly (`ĥ = z + P_{v⊥} r`, a correction orthogonal to `v`)
but is trained against the **downstream language-model loss**, *not* against a manifold-distance
surrogate. It recovers **84%** of the fluency lost at strong steering (`ΔLM` +2.78 → +0.44 nats at
`α=8`; 83.3 ± 2.0% across five training seeds) — and it does so by moving *further* off the statistical manifold, so "on-manifold" and
"safe for the LM" are genuinely decoupled. The takeaway for on-manifold steering methods:

```math
\hat{h} = z + P_{v^{\perp}}\, r_\theta(h, z, v, \alpha)
```

keep this projection-preserving form, but supervise `r_θ` with the **downstream LM objective**,
never a manifold-distance surrogate. The four parts below establish this, test how far it
generalizes, ask whether one corrector can serve many steering directions, and check what happens
when the corrector actually *generates* text.

## How this report is organized

### [Part 1 — The core result: correcting off-manifold steering](REPORT_1_core_correction.md)
*Experiments 2–5, 16, 17.* The central negative-then-positive result. The provably-optimal
**Gaussian-manifold** corrector lowers the off-manifold distance by 22% yet **raises** LM loss to
+4.2 nats (statistical "on-manifold" ≠ "LM-safe"); the **same form trained on the downstream LM
loss** recovers 84% of the fluency at `α=8` by moving *further* off the Gaussian manifold. The
corrector extrapolates past its training strength and is **direction-specific** (no transfer to a
near-orthogonal formality vector, but the recipe reproduces per direction, 83–104%). Experiment 16
shows *why* the Gaussian is the wrong yardstick (the real manifold is low-dimensional ~8–34 ≪ 768,
near rank-1, heavy-tailed); Experiment 17 shows the Cold-Diffusion **corruption model + LM
supervision**, not the iteration count, carry the result.

### [Part 2 — Amortizing the corrector across many steering directions](REPORT_2_amortization.md)
*Experiments 6–9, 14.* The base corrector is one-model-per-vector. A **direction-conditional**
corrector trained on a *bank* of directions is one model that corrects them all (55–70% recovery)
and partially transfers to a held-out direction (51% → 7% from weak to strong steering). But the
held-out gap does **not** close by adding directions (Exp 7, transfer drops), adding parameters
(Exp 8, overfits), or curating the bank *toward* the target (Exp 9, backfires). Bank **diversity**,
not target alignment, drives transfer — confirmed causally by a confound-free swap (Exp 14).
Amortized correction is capped by the **training signal**; the reliable route to an unseen direction
remains a per-direction corrector.

### [Part 3 — External validity: does the fluency result generalize?](REPORT_3_external_validity.md)
*Experiments 12, 13, 19, 21, 24, 15, 18, 26.* The core fluency result holds on seven independent axes:
**layer** (blocks 3/6/9: 90/84/76% at `α=8`), **model size** (GPT-2 medium & large, 89% & 84%;
flat across a 6× parameter range), **architecture** as a *sweep of three families* (Qwen3-1.7B 94%,
Pythia-410m/GPT-NeoX with a parallel-residual block 81% — 81–94% band), **prompt family** (77% on
technical prose, 60% on out-of-distribution code), **steering-vector family** (DiffMean /
logistic probe / PCA-contrast, 84–101%), and **training seed** (five-seed recovery 83.3 ± 2.0% / 88.3 ± 2.2% /
85.1 ± 1.1% / 80.8 ± 1.6% / 94.8 ± 1.6% at `α=8` on GPT-2 small / GPT-2 medium / GPT-2 large / Pythia / Qwen3 —
the headline 84% is reproducible, and the recipe is seed-stable across all five headline models spanning three
scales and two architectures). Off
the Gaussian manifold at every setting.

### [Part 4 — Behavioral reality-check: from fluency to real steering in generation](REPORT_4_behavioral.md)
*Experiments 10, 11, 20, 22, 23, 25.* The `ΔLM` recoveries are teacher-forced at one layer. When
the corrector **generates**, it prevents raw steering's collapse into repetition but its output is
only weakly steered (~one-sixth of raw's effect on GPT-2), because the correction is orthogonal to
`v` but not to the downstream concept readout — matched projection is **not** matched behavioral
steering. This caveat is architecture-robust (Qwen3, 10–29% of raw's effect; milder on Pythia). A
**readout-preservation term** (Exp 11) recovers 2–6× more effect and dominates raw at moderate
steering; a **differentiable-generation** term (Exp 20) pushes the frontier out again; the fix's
mechanism transfers to Qwen3 but its Pareto payoff is gated by whether the raw baseline degenerates
(Exp 23). The strong-effect-and-fluent corner stays genuinely hard for a projection-preserving
corrector.

## Headline numbers at a glance

| Result | Number | Where |
|---|---|---|
| Gaussian-manifold corrector: off-manifold distance ↓ but LM loss ↑ | `D_M` −22%, `ΔLM` +4.2 nats @`α=8` | Part 1 (Exp 2) |
| Learned (downstream-LM) corrector: fluency recovered @`α=8` | 84% (`ΔLM` +2.78 → +0.44) | Part 1 (Exp 3) |
| Corrector moves *further* off the Gaussian manifold | `D_M` above raw at every `α` | Part 1 (Exp 3) |
| Direction-specific; recipe reproduces per direction | 0% transfer / 83–104% native | Part 1 (Exp 5) |
| Real manifold is low-dim, anisotropic, heavy-tailed | intrinsic dim ~8–34, PR 1.1, `D_M²` spread 6.7× χ² | Part 1 (Exp 16) |
| Corruption model + LM supervision, not iteration, carry it | one-shot 84% vs iterative 85% vs generic prior −5% | Part 1 (Exp 17) |
| One conditional corrector serves a bank; partial held-out transfer | 55–70% in-bank; 51%→7% held-out | Part 2 (Exp 6) |
| Bank diversity, not target alignment, drives transfer | curated bank −183% @`α=1` | Part 2 (Exp 9, 14) |
| Layer-robust (blocks 3/6/9 @`α=8`) | 90% / 84% / 76% | Part 3 (Exp 12) |
| Model-robust (GPT-2 small/medium/large @`α=8`) | 84% / 89% / 84% | Part 3 (Exp 13, 19) |
| Architecture-robust as a 3-family sweep @`α=8` | 81–94% (GPT-2 / Qwen3 / GPT-NeoX) | Part 3 (Exp 21, 24) |
| Prompt-family-robust @`α=8` | 77% prose / 60% code | Part 3 (Exp 15) |
| Steering-vector-family-robust @`α=8` | 84% / 84% / 101% | Part 3 (Exp 18) |
| Seed-robust across 5 seeds @`α=8` (small / medium / large / Pythia / Qwen3) | 83.3 ± 2.0% / 88.3 ± 2.2% / 85.1 ± 1.1% / 80.8 ± 1.6% / 94.8 ± 1.6% | Part 3 (Exp 26–30) |
| Behavioral caveat: generated effect vs raw (GPT-2) | ~1/6 of raw's | Part 4 (Exp 10) |
| Readout-preservation term recovers more effect | 2–6×, dominates raw at moderate `α` | Part 4 (Exp 11) |

## Limitations (overview; each part gives the detail)

1. **The manifold is modeled as a single Gaussian**, so `D_M` captures scale/correlation but not
   multimodal or nonlinear structure. Part 1 (Exp 16) quantifies the gap and Exp 17 shows a real
   Gaussian-noise diffusion prior does *not* help — hence the downstream-LM objective, with `D_M`
   reported only as a diagnostic.
2. **`ΔLM` is a teacher-forced fluency proxy at one layer.** Part 4 shows this is not the same as
   matched behavioral steering in generation; behavioral effect must be measured directly.
3. **Held-out cross-direction transfer at strong steering is unsolved** by bank size, model size, or
   target-aligned curation alone (Part 2); a per-direction corrector is still needed.
4. **Open axes:** GPT-2 XL and further architectures (state-space / MoE) for the fluency sweep; a
   richer training objective and diverse-bank composition for amortization.

*Per-experiment methods, tables, figures, observations, interpretations, limitations, and
next-checks live in the four part files linked above. All change history is in CHANGELOG.md.*
