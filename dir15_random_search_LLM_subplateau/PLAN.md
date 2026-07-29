# PLAN — Direction 15: Random search for LLM activation sub-plateaus

> Working folder: `dir15_random_search_LLM_subplateau`. The agent rewrites “Current status” and “Next step”
> and ticks stages after each iteration. Disk (`PLAN.md`, `JOURNAL.md`, `RESULTS.md`, `REPORT.md`,
> `CHANGELOG.md`, `../BUDGET.md`, and `../CLAUDE.md`) is the only memory.

## Motivation and research question

MNIST activation interpolations sometimes followed an `A → C → B` prediction path: after leaving the
decision region of endpoint A, the path remained in a stable third-class region C before entering endpoint B.
Previous LLM experiments used a few hand-picked prompt or token pairs and may simply have missed an analogous
phenomenon. A random, held-out, in-distribution pair screen gives an unbiased estimate of how often these
sub-plateaus occur naturally.

The primary question is: when interpolating between early-layer activations from two natural language examples
with different endpoint next-token predictions, how often does the downstream model exhibit two distinct
transitions with a persistent third-token region between them (`A | C | B`)? The follow-up question is whether
the strongest intermediate regions behave like meaningful model states: do they support coherent continuations,
and are they near activations produced by held-out natural contexts whose next-token prediction is C?

This is a discovery screen, not a search over hand-selected semantic contrasts. The sample, interpolation grid,
layers, and candidate score must be frozen before inspecting candidate identities.

## Primary assay

### Model, data, and hook point

- Use the same GPT-2 Large model/tokenizer revision and patching conventions already validated in
  `../dir14_does_context_matter/`; reuse its code where practical rather than silently creating a different
  assay. Record model/tokenizer revisions, dtype, device, and library versions.
- Draw contexts from a held-out split of the same natural-text distribution used for all conditions. Prefer a
  standard frozen validation split already available locally; if none is cached, use a deterministic slice of
  an established GPT-2-like corpus and record its dataset revision. Do not mix datasets within the primary
  screen.
- For each context, use the final non-padding position and cache `resid_post` at preregistered early transformer
  blocks. The default layer set is blocks 0, 2, 4, and 6; reduce it only for a documented resource constraint,
  never after viewing which layer looks best.
- Define endpoint label A or B as the unpatched model's top-1 next-token prediction for that context. Retain only
  contexts with finite activations and pair endpoints whose top-1 token IDs differ. Dataset labels or observed
  next tokens may be reported, but they do not define A and B.

### Frozen random-pair bank

Before running interpolation, save a manifest containing context text or source indices, token IDs, endpoint
predictions and probabilities, activation norms, pair IDs, layers, and the random seed. Shuffle eligible
contexts once and pair consecutive examples without replacement; reject only same-prediction pairs and continue
through the frozen shuffle. Do not filter for semantic relationship, endpoint confidence, activation distance,
or visually interesting behavior.

Target at least 1,000 valid pairs for the primary screen. If the time or memory budget prevents 1,000, freeze
the largest feasible count before inspecting results, label the run exploratory, and give binomial uncertainty
on the observed candidate rate. Keep a fully separate frozen validation bank for confirming all thresholds and
ranked candidates; no pair may appear in both banks.

### Interpolation and downstream propagation

For each pair and preregistered layer, interpolate the two final-position `resid_post` activations using
directional spherical interpolation with linearly interpolated L2 norm (`slerp_rescale`), matching direction 14.
Use 50 evenly spaced alpha values including both endpoints. Patch only the final sequence position and propagate
the patched activation through every remaining transformer block and the language-model head.

At every alpha, calculate the complete final next-token logit and probability vectors before reducing them to
summaries. Stream these vectors in batches: save full vectors for confirmed candidates and sufficient top-k,
endpoint-token, winner-token, entropy, and divergence summaries for every screened path. Do not keep the entire
pair × layer × alpha × vocabulary tensor in RAM. Preserve raw full-distribution shards if the storage budget
allows; otherwise state clearly that all tokens were evaluated but only candidate vectors and screen summaries
were retained.

Record for every path:

- the top-1 token ID and probability at each alpha;
- probabilities and logits for A, B, and every intermediate top-1 token encountered;
- predictive entropy and Jensen–Shannon divergence between adjacent alpha distributions;
- the argmax run-length encoding and exact transition alpha values;
- endpoint reproduction errors and activation norms.

The primary evidence is the full probability evolution. A top-1 label sequence alone is insufficient because a
one-token run can arise from a near-tie or sampling-grid noise.

## Preregistered `A | C | B` definition and ranking

Orient every path from A at alpha 0 to B at alpha 1. A **third-token segment** is a maximal consecutive alpha run
whose top-1 prediction is the same token C, with `C != A` and `C != B`. A path is a primary `A | C | B`
candidate only if:

1. the reproduced endpoints predict A and B respectively;
2. the sequence contains an A-dominant run, then one C-dominant run, then a B-dominant run in that order;
3. C remains top-1 for at least 3 consecutive alpha points (the default persistence threshold);
4. C exceeds both A and B in probability at every point in the retained C run, with a positive minimum margin;
5. entry into and exit from C occur at distinct adjacent-alpha distribution changes, rather than a single
   A/B crossing with an isolated near-tie.

Extra transient tokens outside the central segment must be reported. Classify a candidate as **clean** only when
the compressed top-1 path is exactly `A, C, B`; otherwise classify it as **complex with a persistent C region**
and do not silently coerce it to `A | C | B`.

Freeze an implementation-level candidate score before opening token strings or continuations. Rank candidates
using a simple combination of C-run alpha width, the minimum probability margin of C over `max(p(A), p(B))`,
and separation between the two transition locations. Report each component, not only the composite. Show
sensitivity to persistence thresholds of 2, 3, and 5 points and to a minimum C-margin threshold; the primary
rate always uses the frozen default.

Estimate the candidate prevalence across random pairs with a binomial confidence interval, both overall and by
interpolation layer. Pair-level prevalence counts a pair once even if multiple layers qualify; layer-level rates
may count layer-paths. Confirm the complete detection and ranking procedure on the untouched validation bank
without retuning it.

## Candidate inspection

Only after the screen and ranking are frozen, inspect the highest-scoring candidates plus a random sample of
threshold-passing candidates. This prevents the report from presenting one unusually legible token triplet as
typical.

### Generated continuations

For each inspected candidate, patch representative points from the A region, center of the C region, and B
region at the same layer. Generate continuations from the downstream distribution with:

- greedy decoding as the deterministic primary view;
- a frozen stochastic decoding configuration and seeds as a secondary view;
- an unpatched continuation from each endpoint context as a control.

Because a patched activation belongs to one endpoint's sequence context, run the same activation under both
endpoint contexts where tensor shapes permit and label the conditioning context explicitly. Do not describe a C
token as a stable semantic state unless continuation behavior is reproducible across neighboring C-region alpha
points and is not an artifact of one decoding sample.

### Nearby natural activations

Build a separate reference bank of held-out natural contexts, disjoint from both pair banks. Cache activations at
the same layer and their unpatched next-token distributions. For each inspected C-region activation:

- retrieve nearest natural activations using cosine distance and norm-normalized Euclidean distance;
- report neighbor source contexts, predicted next tokens, distances, and local label composition;
- compare against distance-matched A- and B-region query points and against random natural queries;
- quantify whether C-region points are unusually close to natural activations whose top-1 prediction is C.

Use exact search if feasible; otherwise validate approximate-neighbor recall on a subset. Fit no reference set
using interpolation points. A nearby C-predicting context is evidence of local similarity, not proof that the
interpolation lies on the natural activation manifold.

## Controls and validity checks

- Confirm alpha 0 and 1 reproduce unpatched endpoint logits and distributions within a stated tolerance.
- Re-run a subset deterministically and compare arrays.
- Run same-context/self-pair interpolations; these should not create a third-token region.
- Run same-endpoint-prediction pairs as a secondary negative control, kept out of the primary prevalence
  denominator.
- Compare against linear interpolation as a secondary geometry control; do not replace `slerp_rescale` in the
  primary assay.
- Verify that batching, padding, and attention masks do not change endpoint predictions.
- Plot probability curves around both transitions and adjacent-distribution divergence so near-ties and
  grid-resolution artifacts are visible.
- Report token-frequency effects: common intermediate tokens may dominate the census without representing a
  coherent third region.

## Success criterion

`RESULTS.md` and a self-contained `REPORT.md` contain:

- a frozen manifest and a random primary screen of at least 1,000 valid in-distribution, different-prediction
  pairs, or a clearly labeled maximum-feasible exploratory screen;
- all preregistered early layers, 50 interpolation points, endpoint-fidelity checks, and the full-distribution
  summaries needed to regenerate every classification;
- overall, pair-level, and per-layer `A | C | B` prevalence with uncertainty, plus clean-versus-complex counts;
- the frozen candidate rule, ranked component scores, threshold sensitivity, and confirmation on a disjoint
  validation bank;
- aggregate figures showing top-1 composition, segment widths, margins, transition locations, and intermediate
  token identities rather than only selected examples;
- detailed probability trajectories and generated continuations for frozen top-ranked and randomly selected
  candidates;
- nearest-natural-activation analysis with appropriate A/B and random-query controls;
- a bounded verdict distinguishing a robust third output region, a fragile top-1 near-tie, and no detected
  LLM analogue of the MNIST phenomenon.

A null result is complete if the screen has adequate endpoint fidelity, the detection power implied by its
sample size is reported, and the validation and controls pass. When all criteria are satisfied and no
unaddressed feedback remains, write an empty `STOP` file.

## Required artifacts

- frozen primary, validation, and natural-reference manifests with seeds and disjoint source indices;
- scripts under `experiments/` and numeric outputs or sharded arrays under `results/`;
- a machine-readable candidate table with raw rule components and provenance;
- `plots/candidate_prevalence_by_layer.*`;
- `plots/segment_width_margin_distribution.*`;
- `plots/intermediate_token_census.*`;
- `plots/top_candidate_probability_paths.*`;
- `plots/natural_neighbor_comparison.*`;
- current-best `RESULTS.md` and self-contained `REPORT.md` with every quantitative plot embedded.

## Stages

- [x] **S1 — Lock protocol and manifests.** Reuse and validate direction 14's GPT-2 Large implementation,
  freeze dataset split, layers, pair construction, thresholds, score, seeds, and disjoint primary/validation/
  reference manifests before viewing interpolation results.
- [x] **S2 — Run the random primary screen.** Interpolate all frozen pairs at every selected layer, stream the
  complete next-token distributions through the detector, save summaries, and verify endpoints and reruns.
- [x] **S3 — Census and validate candidates.** Apply the frozen rule, report prevalence and sensitivity, freeze
  ranked and random inspection sets, and confirm the untouched validation bank without retuning.
- [x] **S4 — Inspect continuations.** Generate controlled continuations from A, C, and B regions for the frozen
  candidate subset and distinguish reproducible behavior from decoding noise.
- [x] **S5 — Compare with natural activations.** Retrieve and quantify nearby held-out natural activations at
  the same layers, including A/B and random-query controls.
- [x] **S6 — Synthesize and finalize.** Run geometry and self-pair controls, curate aggregate and candidate
  figures, write the bounded verdict in `RESULTS.md` and `REPORT.md`, update history, and write `STOP` only after
  checking for unaddressed feedback.

## Fallback if time runs short

Prioritize one fully validated early layer over a shallow, incomplete multi-layer sweep: freeze and screen the
largest feasible random bank, preserve the disjoint confirmation bank, and inspect the top three plus three
random qualifying candidates. Generated continuations come before nearest-neighbor analysis. If no candidate
passes, spend the remaining analysis time on detection power, near-miss distributions, and controls rather than
relaxing the rule post hoc. Reserve the final 20 minutes for current-best figures, `RESULTS.md`, `REPORT.md`,
`CHANGELOG.md`, and the feedback/`STOP` check.

## Out of scope

- No hand-picked pair may enter the primary prevalence estimate.
- No threshold, layer, interpolation method, or sample exclusion may be chosen because it reveals an appealing
  candidate.
- No prompt engineering or semantic-pair benchmark as a substitute for the random in-distribution screen.
- No claim that a third top-1 token alone establishes a natural activation region or semantic concept.
- No model comparison or training-time study until the GPT-2 Large primary assay is complete.
- No installation or replacement of torch, torchvision, TransformerLens, JAX, or Flax.
- Read `../BUDGET.md` and `../CLAUDE.md` every iteration; keep current-best results in `RESULTS.md`/`REPORT.md`
  and history in `CHANGELOG.md`.

## On-track check

End each `JOURNAL.md` entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status

**Complete (S1-S6), plus one operator-feedback iteration.** The protocol was frozen and run: 5,980
32-token WikiText-103-validation windows, three disjoint banks, GPT-2 Large `resid_post` at blocks
0/2/4/6, 50-alpha `slerp_rescale` paths, 8,000 primary paths (+2,400 validation, +6,800 control).
Headline: **16.9% of eligible paths [16.1, 17.8] show a persistent third top-1 token**, replicated at
17.7% on the disjoint validation bank, 0% on self-pairs, 16.1% under linear interpolation and 11.1%
on same-prediction pairs. The typical third region is weak (higher entropy than the endpoints,
generic token, off-manifold, unsupported by natural neighbours); a ~3-4% minority is a genuine crisp
third state.

`human_feedback_1` (worked examples + Matthew-style plateau plots) was addressed and the file renamed
`human_feedback_1.addressed.md`. That added the output-geometry view: relative output distance
`d(α)`, C-window flatness `ρ`, a matched non-candidate control, worked examples with both source
texts and the full top-1 sequence, and three new figures. Key new number: **only 1.39% of eligible
paths [1.15, 1.68] hold a true flat sub-plateau (ρ < 0.5)** — median candidate ρ is 2.05, so a
persistent third token is usually a label event inside the boundary, not a shelf.

An exploratory depth sweep (same pairs, same detector, blocks 12/18/24/30, excluded from the headline)
then answered the plan's own open question: the depth trend **turns over**. The third-token rate peaks
between blocks 6 and 12 and falls to 1.7% by block 30; the sub-plateau rate peaks at block 6 (2.87%)
and is zero from block 18 on. The phenomenon is early-to-mid network. Verdict, all figures and all
controls are in `RESULTS.md` / `REPORT.md`; history in `CHANGELOG.md`.

`human_feedback_2` (*"does the sub-plateau exist in real language data?"*) was then addressed and the
file renamed `human_feedback_2.addressed.md`. It removed the synthetic step entirely: paths built in
**text** space (step k = context B's first k tokens ++ context A's remaining 32−k), so all 33 points
are real 32-token sequences run through the unmodified model. Two frozen banks, 2,000 paths. Key new
number: under one symmetric rule (A, C and B runs each ≥3 points) applied to both, **7.9% of
real-language paths [6.4, 9.7] hold a true sub-plateau against 1.29% of activation paths
[1.06, 1.57]** — six times more common — and the median C-window flatness flips from ρ = 2.05 to 0.45.
This retires the direction's strongest objection: the third region is *not* an artefact of leaving the
activation manifold. The real-language shape is different, though — a many-step staircase (7 top-1
predictions per path vs 3) with sharp boundaries (motion concentration κ = 0.49 vs 0.1 for a ramp),
so the shelf is one step of a long climb, and only 5.8% of real-text candidates are a clean `A, C, B`.
The same iteration fixed a deliverable bug: all figure captions had been in alt text (which never
renders), so both files now carry visible numbered captions and pass `experiments/check_render.py`.

## Next step

None — every success criterion is met and no unaddressed feedback file remains, so `STOP` is written.
If reopened: (a) build the real-language bank R2 from a window pool disjoint from the primary bank so
that screen becomes independent evidence; (b) replace the splice with a morph whose intermediate
points are natural prose rather than spliced prose; (c) generate continuations from real-text C
shelves (the S4 analogue, never run for that section); (d) give the depth sweep its own disjoint pair
bank; (e) enlarge the natural reference bank beyond 2,000 contexts so nearest-neighbour distances are
not inflated by the small search space.
