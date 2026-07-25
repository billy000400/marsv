# PLAN — Do activation plateaus depend on context?

> Working folder: `dir14_does_context_matter`. The agent rewrites "Current status" and "Next step" and ticks stages each iteration. Disk (`PLAN.md`, `JOURNAL.md`, `RESULTS.md`, `REPORT.md`, `CHANGELOG.md`, `../BUDGET.md`, and `../CLAUDE.md`) is the only memory.

## Research question

Matthew's activation-plateau post shows sharp downstream transitions when interpolating between GPT-2 Large residual-stream activations for prompts such as `The house was big` and `The house was large`. This direction asks whether those plateaus are properties of the endpoint words themselves or whether they depend on the preceding context.

Test two complementary interventions:

1. **Fix the context and vary the endpoint pair.** Under the same prefix, compare `big → large` with `big → in`. This tests whether plateau behavior depends on which two endpoint tokens are connected.
2. **Fix one transition and vary the context.** Hold `big → large` fixed while testing four prefix conditions: no context, a random token sequence, natural but unrelated context, and semantically relevant context. This tests whether meaningful context changes the plateau even when the endpoint words are unchanged.

For every condition, use the same activation interpolation and compare plateau presence, plateau width, and transition location. The primary model is **GPT-2 Large**, matching Matthew's post.

## Source-locked primary assay

Use Matthew Shinkle and StefanHex's post and released repository as the implementation source of truth. Before extending the assay, reproduce this reference configuration with GPT-2 Large:

```yaml
model_name: "gpt2-large"
shared_context: "The house was"
token_pairs:
  - ["big", "in"]
  - ["big", "large"]
n_steps: 50
```

For each endpoint pair and interpolation layer:

1. Form the two complete prompts from an identical prefix plus one endpoint token.
2. Collect the final-position `resid_post` activations for both prompts.
3. Apply Matthew's `slerp_rescale`: spherical interpolation of activation direction and linear interpolation of L2 norm.
4. Use exactly 50 evenly spaced interpolation values, including both endpoints.
5. Patch only the final sequence position at the selected `resid_post` layer.
6. Sweep every GPT-2 Large interpolation layer and record Matthew's downstream hooks: `attn_out`, `resid_mid`, `mlp_post`, `mlp_out`, `resid_post`, and final logits.
7. Compute Matthew's relative endpoint distance at each recording site:

```math
d(t)=\frac{\lVert x(t)-x_A\rVert_2}{\lVert x(t)-x_A\rVert_2+\lVert x(t)-x_B\rVert_2}.
```

The raw `d(t)` curves are the primary evidence. A plateau remains near one endpoint, changes sharply, and remains near the other. Do not substitute linear interpolation, a radial perturbation assay, a different model, or a reduced layer sweep for the primary experiment.

## Experiment 1 — fixed context, different endpoint pairs

Use the exact shared prefix `The house was` and compare:

- `The house was big` → `The house was large`
- `The house was big` → `The house was in`

Everything except the second endpoint token must be identical: model weights, interpolation grid, interpolation and recording layers, patch position, hooks, precision, and plotting axes. This is the Matthew reference comparison and directly tests endpoint-pair sensitivity under fixed context.

Because one endpoint (`big`) is shared but the other differs, this experiment does not estimate a generic semantic-similarity effect. It answers the narrower controlled question: does changing `large` to `in` alter plateau presence, width, or location under the same prefix?

## Experiment 2 — fixed `big → large` transition, different contexts

Hold the endpoint transition fixed and evaluate four prefix classes:

1. **No context.** The endpoint token alone. Treat the tokenizer's required leading-space convention consistently and document the exact token IDs and decoded strings.
2. **Random sequence.** A length-matched sequence sampled from GPT-2 tokens. Exclude special tokens and freeze the random seed and sampled sequence before inspecting any plateau curves.
3. **Natural but unrelated context.** A grammatical prefix whose content does not constrain a size adjective at the endpoint.
4. **Semantically relevant context.** A grammatical prefix that naturally makes `big` or `large` a relevant continuation, anchored by Matthew's `The house was` context.

Create a small preregistered bank of prefixes for the random, unrelated, and relevant classes rather than relying on one potentially idiosyncratic sentence. Freeze the exact strings, token IDs, selection rule, and random seed in a config or manifest before running interpolation. Match prefix token length across the three contextual classes wherever possible. If exact matching is impossible without damaging the category definition, report token lengths and include length as an explicit limitation; do not silently truncate activations or compare different patch positions.

For every prefix, construct both endpoints by appending the same single-token forms of `big` and `large`. Assert that the endpoint token IDs are identical across contexts. If GPT-2 tokenization changes an endpoint because of whitespace or boundary handling, correct prompt construction before running; never compare conditions with different endpoint token IDs while calling the transition fixed.

The primary comparison is across the four context classes. Replicates within a class show whether any effect generalizes beyond a particular prefix; they are not a license to select the clearest example after seeing results.

## Measurement and comparison

### Primary evidence

Show raw layerwise `d(t)` curves for every condition using shared axes. Produce paired panels for Experiment 1 and context-class panels for Experiment 2. Preserve individual-prefix curves; class averages alone can hide incompatible transition locations or wash out sharp plateaus.

### Descriptive summaries

Use the same prespecified extraction code for all curves:

- **Plateau presence:** a bounded descriptive classification based on sustained endpoint-near regions on both sides of a monotone transition. Freeze the endpoint-nearness threshold and minimum run length before comparing conditions. Report sensitivity to reasonable threshold choices; do not tune the rule per context.
- **Transition width:** the interpolation interval between the first crossings of `d(t)=0.1` and `d(t)=0.9`, computed after orienting endpoints consistently. Narrower means a sharper transition. If a curve never crosses both levels or is materially non-monotone, report width as undefined and show the raw curve rather than forcing a number.
- **Transition location:** the interpolation value where `d(t)` first crosses `0.5`. If no crossing exists, report it as undefined.

The 50-point grid limits the resolution of width and location. Report values on that grid without implying sub-grid precision. Plateau presence, width, and location are summaries of the raw curves, not replacements for them.

Compare conditions at every interpolation and recording layer. For the compact headline summary, use final logits and report the full layer sweep beside it so a context effect is not inferred from a cherry-picked layer. Summarize replicate prefixes by distributions and paired differences, with bootstrap confidence intervals across frozen prefixes only if the bank is large enough to make them meaningful. Do not treat interpolation points or layers as independent samples.

## Validity checks

- Record the exact GPT-2 Large revision, tokenizer revision, library versions, dtype, device, and random seeds.
- Save every prompt, token ID sequence, and decoded token sequence. Confirm that each endpoint is one token and that the shared prefix tokens are identical within each pair.
- Verify interpolation endpoints reproduce the unpatched endpoint activations and logits within numerical tolerance.
- Verify `d(0)` and `d(1)` reach the expected endpoint values at every valid recording site.
- Re-run a subset deterministically and compare saved arrays.
- Run Matthew's exact reference configuration before interpreting new context conditions. If it does not reproduce the qualitative reference phenomenon, diagnose or report the failed replication rather than drawing a context conclusion.
- Store raw activations or sufficient numeric outputs to regenerate every curve without rerunning GPT-2 Large.

## Success criterion

`RESULTS.md` and `REPORT.md` are complete when they contain:

- a GPT-2 Large reproduction of Matthew's exact `The house was` assay for `big → large` and `big → in`;
- the fixed-context endpoint-pair comparison;
- the fixed-`big → large` comparison across no context, frozen random sequence, natural unrelated context, and semantically relevant context, including the preregistered prefix bank;
- raw all-layer `d(t)` curves and compact final-logit summaries of plateau presence, transition width, and transition location;
- tokenization, endpoint-fidelity, and deterministic-rerun checks;
- a bounded verdict separating endpoint-pair sensitivity from context sensitivity, including heterogeneity across prefixes and layers;
- a self-contained `REPORT.md` with Methods, rendered metric equations, embedded figures, limitations, and no causal or semantic-generalization claim beyond the tested prompts.

Null results are complete if the reference assay and validity checks pass. When all criteria are satisfied and no unaddressed feedback remains, write an empty `STOP` file.

## Required artifacts

- a frozen experiment config and prompt manifest containing all exact strings, token IDs, class labels, seeds, and model/tokenizer revisions;
- raw results for every prefix, endpoint pair, interpolation layer, recording hook, and interpolation point;
- a machine-readable summary of plateau presence, width, transition location, and undefined cases;
- `plots/fixed_context_endpoint_pairs.*`;
- `plots/fixed_transition_contexts.*`;
- `plots/context_effect_by_layer.*`;
- current-best `RESULTS.md` and a self-contained `REPORT.md`.

## Stages

- [x] **S1 — Lock implementation and prompts.** Read Matthew's code, freeze GPT-2 Large and tokenizer revisions, define the prefix bank without viewing results, and save exact tokenization.
- [x] **S2 — Reproduce the reference assay.** Run `The house was` with `big → large` and `big → in`; validate endpoints and the full recording pipeline.
- [x] **S3 — Fixed-context endpoint comparison.** Complete the all-layer paired analysis and plot plateau presence, width, and location.
- [x] **S4 — Fixed-transition context comparison.** Run the frozen `big → large` assay for all four context classes and all preregistered prefixes.
- [x] **S5 — Robustness and synthesis.** Check deterministic replication, threshold sensitivity, prefix heterogeneity, and layer dependence; choose a bounded verdict.
- [x] **S6 — Finalize deliverables.** Curate `RESULTS.md` and `REPORT.md`, embed all necessary figures, document limitations, update history, and write `STOP` only when feedback is clear.

## Fallback if time runs short

Prioritize, in order: the exact GPT-2 Large reference reproduction; the fixed-context `big → large` versus `big → in` comparison; one frozen example from each of the four context classes; then the larger prefix bank and robustness summaries. A reduced prefix bank is exploratory and must not support a broad claim about semantic context. Reserve the final 20 minutes for current-best figures, `RESULTS.md`, `REPORT.md`, `CHANGELOG.md`, and the feedback/`STOP` check.

## Out of scope

- No GPT-2 Small result as a substitute for the GPT-2 Large primary goal.
- No training-time or grokking experiment; this direction studies pretrained GPT-2 Large.
- No replacement endpoint dataset, broad token-pair search, or post-hoc choice of especially clean prefixes.
- No multi-token endpoint workaround presented as the same assay.
- No claim that semantic relatedness causes a plateau difference; the experiment tests sensitivity to the frozen contexts.
- No new plateau-score suite beyond the raw curves and the three requested summaries.
- Do not install or replace the existing CUDA build of torch, torchvision, TransformerLens, JAX, or Flax.
- Read `../BUDGET.md` and `../CLAUDE.md` every iteration; keep current-best results in `RESULTS.md`/`REPORT.md` and history in `CHANGELOG.md`.

## On-track check

End each `JOURNAL.md` entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status

COMPLETE (2026-07-25). All six stages done in one session. Deliverables curated and verified
(8 embedded figures each, GitHub-rendered equations, no KaTeX errors).

Current-best verdict (final logits, patch at block 0): under the fixed prefix `The house was`,
`big → in` gives transition width w = 0.050 (plateau) and `big → large` w = 0.592 (no plateau;
straight-line reference 0.800) — the endpoint pair decides plateau *presence*. With the transition
fixed at `big → in`, sharpness follows a replicated ladder: no context 0.575 >> random tokens 0.141
>> natural English 0.054 (pooled over two disjoint frozen prefix banks, 36 prefixes; exact rank-sum
p = 8e-7 for natural vs random), with topical relevance a weak add-on (0.049 vs 0.063, p = 0.045).
Validity checks pass (endpoint fidelity ~1e-4 absolute, deterministic and cross-script re-runs
bit-identical). Endpoint geometry co-varies with width (Spearman rho = +0.49, p = 0.09) and is
reported as an open confound.

Deviation from the preregistration, logged: Experiment 2 was also run for `big → in`, because the
preregistered pair `big → large` shows no plateau in any context and a context effect on plateaus
cannot be measured on a floor. Both pairs are reported. Bank 2 was frozen and declared before it ran.

## Next step

Done — direction complete and `STOP` written. If new operator feedback arrives, delete `STOP`,
address it, and re-check before rewriting `STOP` (CLAUDE.md rule 11). Natural follow-ups, none
required by this plan: a second model to test whether the natural-vs-random ladder is GPT-2-specific,
and a design that decorrelates endpoint geometry from context class.

## Primary references

- Matthew Shinkle and StefanHex, *Activation Plateaus: Where and How They Emerge*: https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge
- Matthew's released configuration and code: https://github.com/MShinkle/activation_plateau_mechanisms
