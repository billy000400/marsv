# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-10-09 — iteration 1 (start): assumptions logged before coding

Assumptions (ordinary implementation details; no human available):

- **"Original interpolation rule"** = the project-standard rule used in dir22/dir23 (`slerp_lerp_norm`): shortest-arc SLERP of
  the direction, L2 norm interpolated linearly. Applied to the layer-0 block output (residual stream after block 0) at the
  country token, 41 points, Japan (t=0) to Germany (t=1). Rejected: plain linear interpolation (not the rule earlier
  directions used).
- **Bridge position under the path.** Only the country position is patched (plan wording). The ` is` token's own layer-0
  output still comes from a Japan-token base prompt, so t=1 is not bit-identical to the native Germany run; the mismatch is
  measured and reported instead of also interpolating the bridge position.
- **Tuning vs reserved prompts.** Three capital templates are correct for both countries in dir30's filter. Tune on
  "The capital city of {C}" + " is" (different prefix from the original); reserve the original "The capital of {C}" + " is"
  and "The capital of {C}" + " is the city of". Rejected: tuning on the "is the city of" paraphrase, because it shares the
  original's prefix and therefore its country-position state.
- **Frozen donors.** Donor activations are the tuning pair's opposite-country values, stored once and reused on reserved
  prompts, context prompts and the whole path.
- **Country readout.** Mean-difference linear readout (Japan vs Germany) on 16 explicit-country non-capital prefixes per
  country in four template families; validated leave-one-family-out. Rejected: logistic regression (2 classes, n=32,
  d=1024 separates trivially; the mean-difference direction has no hyperparameter).
- **Random controls.** Same layer, position, feature count and coefficients, placed on random dictionary decoder
  directions and rescaled to the real edit's L2 norm (three seeds). Rejected: random *active* features, because only
  about 7 features are active per layer at this token, too few to draw 10 from.

### S1 done; selection rule for S2 fixed BEFORE the scan is run

S1 (results/s1_plateau.log): all six capital prompts answered correctly; all tokens single. Path t=0 reproduces the native
Japan run (max logit diff 4e-5); t=1 differs from the native Germany run by at most 0.13 logits (P(Berlin) 0.4776 vs
0.4770), because the ` is` token's own layer-0 output comes from the Japan-token base (11.6% relative difference there).
Country-position hidden states: adjacent-step profile is flat at layers 0-9, and becomes peaked at t~0.51 in layers
20-27 (layer 26: peak step 2.65x the mean, edge steps 0.45x). Top next token at the country position is ` is` at all 41
points. After the bridge, P(Tokyo) stays ~0.40 until t~0.40 and is ~0 by t=0.60; margin crosses zero at t=0.531.
Readout: leave-one-family-out accuracy 1.0 at every layer.

Selection rule (declared now, before any intervention result):
1. Candidates are country-position edits at layers <= 26 with beta <= 1 (natural donor replacement). beta = 2 is
   extrapolation and is considered only if nothing succeeds at beta <= 1.
2. Success on the tuning pair = the Tokyo-minus-Berlin logit margin changes sign in BOTH directions at the same
   layer(s), feature count and beta.
3. Smallest = fewest features in total (k x number of layers), then smaller beta, then the larger worst-direction
   margin shift.
4. If nothing at the country position reverses the margin even at beta = 2 with two adjacent layers, the gate fails and
   the study stops at S2 with that limitation.

### 2026-10-09 — iteration 1 (end): S2, S3, deliverables

What I did
- `s2_slices.py`: one pass over all 28 transcoder layers, keeping only rows of features active at the country or ` is`
  token of the tuning/original pairs plus 64 random decoder rows per layer (25 dictionaries streamed through /tmp).
- `s2_scan.py`: gradient-projection ranking on the tuning pair and real edits (k = 1/5/10, beta = 0/0.5/1/2, both
  directions) at the ` is` token (layers 0-27), the country token (layers 0-26) and adjacent-layer pairs. 1,944 runs.
- `s2_select.py`: applied the rule declared above. Four settings reverse the margin in both directions, all at beta = 2
  and all including layer 16. Selected layer 16, top-10 features, beta = 2 (fewest features).
- `s3_path.py`: frozen edit on reserved prompts at beta 0.5/1/2 with 3 random controls, on dir30's context prompts, and
  on the 41-point path (baseline, two directed edits, six controls). Also counted how often each frozen feature is
  active on the 32 non-capital readout prefixes.
- `s4_plots.py`: three figures. Wrote RESULTS.md and REPORT.md; `check_render.py` passes on both.

What I learned
- Positive control is partial: ` is`-token edits shift the margin by at most 4.4 (beta = 1) / 8.2 (beta = 2) of ~9.3 needed.
- Layer 16 at the country token is the only strong site (7.06 / 4.13 at beta = 1; next best 2.77 / 1.89).
- The frozen edit is a country swap: readout L26 +1.05 -> -0.84 and -1.21 -> +0.09; currency/language answers flip on
  6/6 mid-sentence prompts; 8 of the 13 distinct frozen features are active on 16/16 non-capital prefixes of one
  country and 0/16 of the other.
- Path: answer boundary and readout crossing vanish under either edit; layer-26 largest step 16.0 -> 5.5 / 6.2; controls
  14.3-19.8. Everything moves together, so the plan's verdict row is "not disentangled".

Decisions and caveats
- The plan gates S3 on a usable edit. I ran the path anyway and report it as a description of the inseparable case;
  the plan's own interpretation table has a row for it. No strength above beta = 2 was tried (plan: do not increase
  strength indefinitely).
- Four context prompts start with the country word. The edit does nothing there and the readout returns ~+1.3 for both
  countries, so they are listed in RESULTS.md and excluded from the count. dir30 saw the same first-token anomaly.
- Not done, by design: single-feature breakdown of capital vs currency effects, other country pairs, attention edits.
  They are outside the plan and the success criterion is already met.
- Unrelated observation, not changed: PLAN.md contains mis-encoded dash characters from its creation.

Revised next step: none; STOP written (no unaddressed feedback files present).

On track? yes — complete, 100% done, no blocker (verdict: not disentangled).
