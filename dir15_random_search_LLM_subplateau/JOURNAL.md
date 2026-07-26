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
