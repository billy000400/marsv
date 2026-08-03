# JOURNAL — Direction 15: Random search for LLM activation sub-plateaus

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-26 — iteration 1: full protocol frozen, screened, analysed, reported (S1-S6)

**Did.** Built the whole assay from scratch and ran it end to end.
- `experiments/common.py` — GPT-2 Large runner with a final-position `resid_post` patch hook,
  `slerp_rescale` (copied from dir14), KV-cached generation, the frozen `A|C|B` detector and Wilson
  CIs, with self-tests (synthetic A|C|B accepted, single A->B crossing and 2-point run rejected).
- `experiments/manifest.py` — 5,980 non-overlapping 32-token windows from the WikiText-103 raw
  validation split; one seed-0 shuffle into three disjoint pools; 1,000 primary pairs, 300 validation
  pairs, 2,000 reference contexts; cached final-position activations at blocks 0/2/4/6 and the
  unpatched top-1 for every window.
- `experiments/screen.py` — 8,000 primary paths (+2,400 validation, +2,400 same-prediction, +2,400
  self, +2,000 linear) at 0.12 s/path by batching 4 paths x 50 alphas per forward.
- `experiments/analyze.py`, `inspect_cands.py`, `plot_inspect.py`, `checks.py` — census, sensitivity,
  continuations, nearest-natural retrieval, determinism/batching checks, 8 figures.

**Frozen assumptions (loop mode: recorded, not asked).**
1. *Conditioning context.* Two random contexts are different token sequences, so the patched
   activation must be propagated inside *some* context. Chosen: run every pair twice, once under each
   endpoint's own tokens, and define A and B as the actual top-1 at alpha 0 and 1 in that context.
   Rejected: (a) a single fixed conditioning context (asymmetric, and half the endpoints would not
   reproduce); (b) restricting to pairs sharing a 31-token prefix (that is dir14's minimal-pair design
   and is not a random in-distribution screen). Consequence measured, not hidden: the foreign endpoint
   reproduces its home prediction on only 17.6% of paths, so the rate is also reported on the
   transfer-consistent subset (14.0%).
2. *Condition 5 of the rule* ("two distinct transitions") implemented as JSD > 0.005 bits at both
   boundaries of the C run. Rejected: comparing against the path's median JSD (data-dependent, i.e.
   not freezable before seeing curves).
3. *Score* = margin_min x transition separation, frozen before decoding any token string.
4. *Corpus.* WikiText-103 validation (downloadable, standard, held-out). Rejected OpenWebText (not
   cached, too large for the time budget).
5. Full 50,257-dim distributions are computed at every alpha but only summaries retained (~125 GB
   otherwise); stated in REPORT.md Methods.

**Learned.**
- The phenomenon is common at the label level (16.9% of eligible paths) and replicates on a disjoint
  bank (17.7%) — so the earlier hand-picked-pair LLM experiments were not missing a rare event.
- It is mostly weak. C regions are 3-5 of 50 grid points wide, *higher* entropy than the endpoints
  (6.97 vs 5.70 bits), headed by ' the' / '.' / ' of', and 11.1% of same-prediction pairs also produce
  one — i.e. much of it is "the interpolant flattens onto a frequency default", not a third opinion.
- Geometry is not the cause (linear 16.1% ≈ slerp 16.9%); the detector is not the cause (self-pairs 0).
- Depth matters a lot: 8.2% at block 0 rising to 27.7% at block 6.
- The neighbour analysis was the most informative negative: C-region points are the *furthest* from
  the natural bank and their neighbours agree with C only 4.5% of the time (14.1% for a natural query).
- Bug caught before publication: the own/foreign endpoint labels were swapped for the
  context-B conditioning, giving a nonsensical 52.4% "endpoint match". Recomputed per conditioning:
  100% own-endpoint match, 17.6% transfer. Only corrected numbers were ever written to a deliverable.

**Next step.** The plan's success criteria are all met and no unaddressed feedback file exists, so
`STOP` is written. If this direction is reopened, the two highest-value extensions are (a) later
interpolation blocks (the block-0->6 trend suggests the rate keeps climbing, and the plan deliberately
froze early blocks only) and (b) a much larger reference bank so the nearest-natural distances are not
inflated by a 2,000-context search space.

On track? yes — S1-S6 complete, 100% of the plan's success criteria met, no blocker.

## 2026-07-26 — iteration 2: operator feedback addressed (worked examples + plateau-style curves)

**Feedback file.** `human_feedback_1.txt` (no `.addressed.md` suffix, so it was this iteration's
work): *"Can you show some examples that sub-plateau shows up? what the sequence? interpolate from
where to where? Show plots in Mathew's plateau lesswrong post style."* Renamed to
`human_feedback_1.addressed.md` after all four asks were done. (The file is `.txt`, not `.md`; I
treated it as feedback anyway — the rule's intent is clear — and renamed it to the required
`.addressed.md` ending.)

**Did.**
- `experiments/matthew_examples.py` — re-ran all 1,290 primary candidates plus 1,290 seed-13 matched
  eligible non-candidates (2,580 paths × 50 alphas, 289 s) keeping the full final-logit vector, and
  computed Matthew's relative output distance `d(α)`, the C-window flatness `ρ`, `d̄_C` and
  `w(10→90)`. No new pairs, no threshold changed, nothing in the primary screen re-derived.
- `experiments/plot_matthew.py` — three figures + `results/matthew_gallery.json`.
- Deliverables: new Methods subsection (equations for `d`, `ρ`, the matched control, `w`), new
  Results section 4 with the worked examples (full 32-token context A and B, endpoint tokens, the
  complete top-1 RLE sequence with alpha ranges, the C-region continuation, the geometry), the
  sub-plateau statistics, and updated Summary / Headline / Conclusion / limitations.

**Learned.**
- **The plateau lens splits the result cleanly, and it is a partly negative answer.** The third token
  is a common *label* event (16.9% of eligible paths) but usually not a *plateau*: median
  ρ = 2.05, i.e. inside the C run the output travels twice as fast as the no-plateau diagonal. Only
  8.2% of candidates (1.39% of eligible paths, CI [1.15, 1.68]) are flat enough (ρ < 0.5) to be
  genuine sub-plateaus. The right sentence is "a sub-plateau shows up on about 1 path in 72", not
  "1 in 6".
- The matched control matters: a random window on an ordinary A→B path is *flatter* (median 1.09) than
  a C run, because it is usually far from that path's single boundary. Without it, ρ = 2.05 could have
  been mis-sold as evidence either way.
- **Validation of the pre-registration.** The candidate score (margin × transition separation), frozen
  before any curve was drawn and never given output geometry, ranks candidates by flatness anyway:
  median ρ 2.65 → 0.93 across score deciles, Spearman −0.34.
- Depth again: 55 of the 106 sub-plateaus sit at block 6, the deepest block preregistered.
- Honest tension worth keeping: the *flattest* shelves are headed by punctuation (`'.'`, `','`) while
  the most semantically striking example (`A=' which'`, `C=' if'`, `B=' her'`, the C region writing
  *"if not all of her crew survived"*) is only the 20th-percentile of flatness at ρ = 0.16. Flat in
  output space ≠ interesting as a token.

**Assumptions (loop mode).** (1) "Matthew's post style" = the `d(t)`-vs-interpolation-coefficient
curve with the no-plateau diagonal reference, as already implemented in
`../dir13_plateau_on_grok_gpt/`; I reused that direction's exact metric definition so the two
directions are comparable. Rejected: inventing a new stylistic format. (2) The ρ < 0.5 "sub-plateau"
cut is post hoc — used only to select the illustration gallery and to quote a tail rate, never to
change the frozen prevalence rule; stated as such in Methods, Results and Limitations. (3) Examples
shown = the six pre-frozen inspection paths first (so the reader sees typical alongside best), then
the post-hoc flattest six, clearly labelled.

**Next step.** No unaddressed feedback remains and every plan criterion is met, so `STOP` is written.
If reopened: (a) later interpolation blocks — both the label rate (8.2% → 27.7%) and the flatness
(median ρ 2.52 → 1.54) improve monotonically to block 6, the deepest preregistered, so the real
sub-plateau rate may be much higher at block 12+; (b) a larger natural reference bank than 2,000
contexts.

On track? yes — S1-S6 complete plus the feedback iteration, 100% of the plan's success criteria met,
no blocker.

## 2026-07-26 — iteration 2 (continued): exploratory depth sweep, blocks 12–30

**Did.** With budget left after the feedback work, ran the extension the plan itself named as the
highest-value reopening step: `experiments/depth_extension.py` screened the same 1,000 primary pairs
with the same frozen detector at blocks 12/18/24/30 (8,000 paths, 935 s) and recomputed d(t)/ρ for
every candidate; `experiments/plot_depth.py` produced `plots/depth_sweep.png`. Both deliverables gained
a clearly-labelled exploratory section (it reuses the primary pairs, so it is not independent evidence
and feeds no headline).

**Learned — the trend turns over, which I did not expect.** Extrapolating the preregistered window
(8.2% → 27.7% from block 0 to 6) would predict an even higher rate deeper in. Instead the third-token
rate peaks between blocks 6 and 12 and falls to 1.7% at block 30, and the true sub-plateau rate peaks
at block 6 (2.87% of eligible paths) and is *exactly zero* at blocks 18/24/30 (0 of 269 / 118 / 33
candidates). Median ρ drifting to 1.24 at block 30 is not flattening — it is the whole output curve
collapsing onto the no-plateau diagonal, because a patch at block 30 leaves only six blocks to fold the
interpolant into a discrete state. The preregistered early window happened to bracket the interesting
region.

**Also worth recording:** the clean-`A,C,B` share of candidates *rises* with depth (22 → 45% from block
12 to 30) while the plateaus disappear. A tidier top-1 label sequence is not a stronger plateau — a
good reminder that the label-level rule and the geometry measure different things, which is exactly the
point the operator's feedback pushed the report towards.

**Assumption (loop mode).** Block set {12, 18, 24, 30} chosen as an even spread of the remaining depth
and fixed before running; rejected sweeping every block (≈9× the compute for a smoother curve that
would not change the conclusion).

**Next step.** None — plan complete, no unaddressed feedback, so `STOP` is written. If reopened: a
fresh pair bank for the depth sweep so it becomes independent evidence, and a larger natural reference
bank than 2,000 contexts.

On track? yes — S1-S6 complete, feedback addressed, depth extension done, 100% of the plan's success
criteria met, no blocker.

## 2026-07-29 — iteration 3: operator feedback #2 — sub-plateaus in real language data

**Feedback file.** `human_feedback_2.txt`: *"Can you check if subplateau exsist in real language
data?"* No `.addressed.md` suffix, so this was the iteration. Renamed after the work.

**Reading of the question (loop mode: recorded, not asked).** The screen already *draws its contexts*
from real language (WikiText-103), so "real language data" cannot mean the corpus. It must mean the
**path**: every intermediate point so far is a patched synthetic activation, and iteration 1's
neighbour analysis had shown those points sit further off the natural manifold than the endpoints.
So the question I answered is: *does the sub-plateau still appear when every point of the path is a
real token sequence run through the unmodified model?* Rejected alternatives: (a) sliding a window
through continuous prose — there is no A/B pair and hence no `A|C|B` question; (b) chaining nearest
natural activations from A to B — 5,980 windows sample 1280-dim space far too sparsely for the chain
to be smooth; (c) re-running the existing screen on a second corpus — that answers "does it replicate
on other text", which is not what was asked.

**Did.** `experiments/real_text_paths.py`: path step k = context B's first k tokens ++ context A's
remaining 32−k, k = 0..32, all 33 points real GPT-2 inputs, no hooks. Two frozen banks (2,000 paths,
~2.5 min): R1 = the same 1,000 primary pairs; R2 = 1,000 new pairs sharing their 32nd token (seed 15).
Reused the frozen detector, ρ, d̄_C, w(10→90); added motion concentration κ. Added a **symmetric**
rule (A, C and B runs each ≥3 points) and re-scored the activation screen with it so the comparison
is like-for-like. `experiments/plot_real_text.py` → two figures.

**Learned.**
- **The answer is yes, and it is stronger in real language.** Sub-plateau rate 7.9% [6.4, 9.7] on
  real-text paths vs 1.29% [1.06, 1.57] on activation paths under the same symmetric rule. Median
  C-window flatness flips from ρ = 2.05 to 0.45; the share of candidates with ρ < 0.5 goes 8.2% →
  55.6%. The matched non-candidate control moves far less (1.09 → 0.58), so this is not "everything
  is flat on a text path".
- **This retires the strongest objection to the whole direction.** The off-manifold result (Section 7)
  had left open that the third region was an artefact of the gap between real activations. It is not:
  removing the synthetic step makes the shelf *more* common, so activation interpolation understates
  rather than manufactures the phenomenon.
- **But the shape is different.** A real-text path visits 7 distinct top-1 predictions (activation: 3)
  and w(10→90) covers 90% of the path. κ = 0.49 (a smooth ramp would give 0.1) says the boundaries
  are still sharp — there are just many of them. So real language gives a **many-step staircase**;
  the sub-plateau is one step of it, and only 5.8% of real-text candidates are a clean `A, C, B`.
- **Text space is dominated by its final token, and that had to be designed around.** On the
  unrestricted random-pair bank, context B's prediction first becomes top-1 only at the *last* step on
  90.8% of paths — the step where the predicted-from token switches — so the symmetric rule fires on
  0.6% of those paths. Matching the 32nd token (bank R2) removes the discontinuity entirely and is the
  bank that answers the question. Reported, not hidden; R2's non-uniform sampling is in Limitations.
- **Deliverable bug found and fixed:** all 24 figure captions in REPORT.md/RESULTS.md lived in the
  alt text, which GitHub never renders — both files were showing unlabelled images (the same failure
  CLAUDE.md rule 12 records from dir13). Added `experiments/add_captions.py` (idempotent) and copied
  dir13's `check_render.py`; both files now have visible numbered captions, short alt text, sequential
  numbering and a by-number citation from the prose, and pass all of rules 8a–8c and 12.

**Next step.** None required — the plan's success criteria were already met and this feedback is now
addressed, so `STOP` is written. If reopened: (a) build R2's pairs from a *disjoint* window pool so
the real-language screen is independent of the primary bank; (b) replace the splice with a
paraphrase-style morph (or a natural document trajectory with A/B anchors) so intermediate points are
natural prose, not spliced prose; (c) generate continuations from real-text C shelves, the analogue of
S4, which was never run for this section.

On track? yes — S1-S6 complete, feedback #1 and #2 both addressed, 100% of the plan's success
criteria met, no blocker.

## 2026-08-03 — iteration 4: operator feedback #3 — token→token interpolation in a fixed context

**Feedback file.** `human_feedback_3.txt` (no `.addressed` suffix, so this was the iteration):
*"It looks like you were interpolating from one context to another context? I was looking for
interpolating from one token to another token with the same context inducing a plateau. Can you
redesign your experiment and search again?"* Renamed `human_feedback_3.addressed.md` after the work.

**Reading of the question (loop mode: recorded, not asked).** The operator is right about the design:
every screen so far moved between two different 32-token passages — the activation screen slerps two
contexts' activations, and even the real-text screen rewrites the context token by token. The
requested design fixes the context and varies only the token being predicted from. I implemented it as
`S_A = c ++ [t_A]`, `S_B = c ++ [t_B]` with a shared 31-token `c`, and interpolated the **token
embedding** as the primary hook (the literal token→token path), plus `resid_post` at the four
preregistered blocks so the new screen is directly comparable with the old one. Rejected alternatives:
(a) interpolating only at block 6 — cheaper but it hides that the shelf lives at the embedding, which
turned out to be the main finding; (b) hand-choosing semantically contrasting token pairs (e.g. ' cat'
vs ' dog') — that is exactly the hand-picked-pair design this direction exists to avoid; (c) walking
through real vocabulary tokens only — kept, but as a control, because it turns out there are no real
tokens in between at all.

**Did.** `experiments/token_interp.py` (1,000-pair frozen bank, seed 21; 5 hook points; controls:
lerp, same-prediction, self-pairs, nearest-real-token) → 27 min on the shared GPU;
`experiments/token_continuations.py` (20-token greedy decodes from 5 path points of all 72
candidates); `experiments/plot_token.py`, `experiments/plot_token_cont.py` → 3 figures. Added
`Runner.forward_embeds` to `common.py` (the only change to frozen code).

**Learned.**
- **The phenomenon does not need a context change.** With 31 of 32 tokens identical at both ends,
  7.2% [5.8, 9.0] of paths hold a persistent third prediction and **1.70% [1.06, 2.71] hold a true
  sub-plateau — more than the 1.34% of the whole-context screen**, and 0 of 72 matched control windows
  qualify. This kills the deflationary reading of the whole direction (that a third prediction is what
  the model does when fed a blend of two passages).
- **A token swap is a near-step function.** Median transition width 0.103 of the path against 0.459
  for a context swap; κ = 0.83. That is why these are the cleanest staircases in the direction — and
  why flatness alone stopped working as a criterion: 43% of *arbitrary* windows on these paths are
  flat. I added the height condition (0.2 < d̄_C < 0.8); it is nearly inert on context paths
  (1.39% → 1.34%) and decisive on token paths.
- **Depth trades shelves for labels.** Interpolating the same token pairs at blocks 0/2/4/6 raises the
  third-token rate to 19.5% but drops the sub-plateau rate to zero, because the boundary widens from
  10% to 42% of the path. The label statistic and the geometry statistic genuinely measure different
  things; only the geometry one tracks the MNIST picture.
- **There is nothing between two tokens in the vocabulary.** Snapping every interpolant to its nearest
  real token gives a median of 2 distinct tokens per path and 0/500 third regions. The shelves are
  reachable by activation editing and by no prompt — an honest limit that also sharpens the safety
  reading.
- **Many shelves are a real state, not an argmax coincidence.** A/B-region points reproduce their
  unpatched endpoint continuations exactly (median 20/20 tokens — a strong extra correctness check),
  while the C run agrees with itself on a median 11 of 20 tokens for true sub-plateaus (29% all 20) and
  shares 0 with either endpoint. The distribution is bimodal: prefix 1 or prefix 20.

**Next step.** None required: the plan's success criteria were already met and this feedback is now
addressed, so `STOP` is written. If reopened: (a) give the token screen a disjoint confirmation bank
(it currently draws from the same 5,980 windows as the primary bank and has no validation half);
(b) restrict token pairs to whole-word tokens, since corpus position 32 is often a sub-word fragment;
(c) run the nearest-natural-activation analysis for token-path C regions; (d) repeat the token screen
on a second model to see whether the embedding-level shelf is architecture-specific.

On track? yes — S1-S6 complete plus feedback #1-#3 all addressed, 100% of the plan's success criteria
met, no blocker.

**Addendum (same iteration).** With time left after the deliverables, I closed the one gap the new
screen had against the plan's own success criteria — no disjoint confirmation bank —
(`experiments/token_validation.py`, 300 pairs from the 3,980 windows the primary token bank never
touched, seed 22, nothing retuned). It replicates: 7.7% [5.2, 11.2] third-token vs 7.2% [5.8, 9.0],
2.00% [0.92, 4.29] true sub-plateau vs 1.70% [1.06, 2.71], and both shape statistics (w = 0.107,
κ = 0.82) land on the primary values. Figure 1(A) now carries the validation group.
