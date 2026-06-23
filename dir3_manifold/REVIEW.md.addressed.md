# Review of Direction #3 Manifold Work

Short verdict: this folder is work on direction #3, "Manifold characterization of
the residual stream", not direction #9, "Plateaus and adversarial examples". If
the intent was to research OOD/adversarial plateau detection, this work is pointed
at the wrong direction. The direction #9 work appears to live in
`/mars-vol/marsv/dir9_ood`.

The raw experiments are useful as a pilot, but the written conclusion is much
stronger than the artifacts support.

## Main Misunderstandings / Overclaims

1. The report treats an AE reconstruction elbow as strong evidence of an
   intrinsic manifold dimension. It says the AE and ID estimators "converge on
   ~12-16" and calls this "strong evidence". That is too strong: the AE was only
   trained for 1200 CPU steps per bottleneck size, and undertraining can easily
   create a fake elbow.

2. The report says the AE curve "flattens" after `k≈16`. The raw table does not
   really show that. The curve keeps improving through `k=256`; in particular,
   `k=32 -> 48` improves more than `k=16 -> 24`, and `k=64 -> 128` also improves
   materially. So `k=16` is a fragile kneedle heuristic, not a clear plateau.

3. It concludes that GPT-2's layer-6 residual stream is a "low-dimensional curved
   manifold". The experiments show low local ID estimates under TwoNN/MLE on one
   pooled activation sample, but they do not establish a well-learned manifold.
   "Curved manifold" is an interpretation, not a demonstrated result.

4. It did not match AE parameter count across bottleneck sizes. The pasted
   research direction references deep autoencoders "matched on parameter count,
   training time, and data" while varying bottleneck dimension. This
   implementation varies only `k` in a fixed architecture
   `768 -> 512 -> 256 -> k -> 256 -> 512 -> 768`, so parameter count also varies.
   Approximate parameter counts range from about 1.05M at `k=2` to about 1.18M at
   `k=256`.

5. It pooled all token positions together. That may be acceptable for a quick
   pilot, but it confounds token identity, token position, document source, and
   residual geometry. The conclusion should be about this pooled-token activation
   sample, not "the GPT-2 residual stream" in general.

## False or Unsupported Claims

These are the places where the writeup is not just overconfident, but wrong or
unsupported by the saved artifacts.

1. Claim: "Per-dimension standardization changes the estimate by <2 everywhere."

   This is false. In `results/id_nonlinear.json`, layer 11 at `n=50000` changes
   substantially:

   - TwoNN: `16.76` centered -> `11.10` standardized, change about `5.66`
   - MLE: `12.89` centered -> `16.32` standardized, change about `3.43`

2. Claim: "TwoNN and MLE agree within ~3 units everywhere."

   This is false. For layer 11, standardized, `n=50000`, TwoNN is `11.10` and MLE
   is `16.32`, a gap of about `5.22`.

3. Claim: nonlinear ID is "an order of magnitude below the linear d95 (94-479)"
   across layers.

   This is false as written. The PCA table has linear `d95 = 6` for layer 3 and
   `d95 = 5` for layer 11, where the nonlinear estimates are larger, not an order
   of magnitude smaller. The claim is only defensible for some layers, especially
   layer 6.

4. Claim: "Estimators validated on synthetic Gaussians."

   This may have been done interactively, but I found no saved validation script,
   validation JSON, or validation mode in the current `experiments/id_estimate.py`.
   The saved work does not substantiate the claim.

5. Claim/caveat: `STEPS=1200/k`.

   This is wrong. `experiments/ae_sweep.py` uses fixed `STEPS = 1200` for every
   bottleneck size.

## What I Would Trust

The PCA table, TwoNN/MLE output, and AE FVU table appear to match the saved JSON
artifacts. I would trust them as pilot measurements, not as a final result.

A more honest conclusion would be:

> On one pooled FineWeb activation sample from GPT-2 small, TwoNN/MLE give low
> local ID estimates around 12-16 at layer 6, and a short-trained AE sweep has a
> weak kneedle elbow near k=16. This is suggestive, but not strong evidence that
> the residual stream is a 12-16-dimensional manifold.

## Suggested Follow-Up

Before treating the result as real:

1. Re-run AE sweeps with longer training and multiple random seeds.
2. Match parameter count across bottleneck sizes, or explicitly report that
   parameter count is not controlled.
3. Run the AE sweep on standardized activations and/or with the massive
   activation dimension removed.
4. Save synthetic validation artifacts for TwoNN/MLE.
5. Report token-position-stratified ID estimates instead of only pooled-token
   estimates.
6. Tone down the conclusion unless those checks still support `k≈16`.
