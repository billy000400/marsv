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

---

## 2026-10-11 — iteration 2 (start), stage S4: assumptions and selection rule logged BEFORE any intervention

Task: operator feedback `human_feedback_additive_berlin.md` (manifest in `.tasks/`, state `in_progress`). Question: can an
additive intervention at the country token promote the later Berlin answer while preserving Japan-related information?
This session is driven over ssh by an orchestrator standing in for `run.sh`; it audits code and numbers before
RESULTS.md / REPORT.md are written, so this phase covers S4.1-S4.4 only. Up to this point I have only read existing
files. No edited forward pass has been run.

Facts read from existing files (to be re-asserted by the new scripts)
- dir30 `feature_sets.json`: the Berlin set has 85 features and is the same at top10/top20/top50 (every one is active
  on both Berlin discovery facts). The Germany set has 204 / 271 / 275 features.
- Berlin features outside the Germany set, layers 1-26: 69 with Germany top50, 69 with top20, 70 with top10. They sit in
  layers 3-26; layers 1, 2 and 5 have none. Two more Berlin-only features are at layer 27, outside the requested range.
- Berlin discovery support is small: 7 prompts from 2 facts (3 "largest/biggest city in Germany" prompts, 4 "Deutsche
  Reichsbahn (GDR) headquarters" prompts). The capital fact is dir30's held-out fact and is not among them. dir30's
  extraction position is the last prompt token (` is` or ` of`).
- Tuning prompt tokens: `The| capital| city| of| Japan| is`, country position 4. Unedited: P(Tokyo) 0.1835,
  logit(Tokyo) - logit(Berlin) 9.53, P(` is` after the prefix) 0.409, layer-26 readout +1.062 (Germany prefix -1.205).
- dir31's decoder rows have unit L2 norm, so a single-feature edit has L2 norm beta x amplitude.

Assumptions (ordinary implementation details; rejected alternative in brackets)
1. Germany exclusion set = dir30's broadest Germany set, top50. [top20, dir30's main cutoff: same 69 candidates. top10:
   70, it would admit one feature that dir30 ranks 11-20 for Germany.] Candidate counts under all three cutoffs and per
   layer are saved. Set exclusion is a filter only; I also save each candidate's dir30 activation rate on the Germany
   and Japan discovery prompts so the reader can see how "outside the Germany set" looks in numbers.
2. Reference amplitude. Encode the layer's native MLP input (ReLU(x W_enc + b_enc), dir30's `encode`) at the last
   prompt token of each of the 7 Berlin discovery prompts; amplitude = median over the prompts where the activation is
   > 0, pooled over prompts. Saved per feature: positive prompts out of 7, facts out of 2, token index and token text of
   each positive prompt, min and max positive activation. Cross-check against dir30's stored `results/acts`.
   [Fact-averaged mean including zeros, dir30's `mean_act_Berlin`: not what the feedback asks. Max: an outlier.
   Anything from the Germany token of a capital prompt: forbidden by the feedback and not computed at all.]
3. The edit. A fixed vector `beta * sum_i amplitude_i * W_dec[i]` is added to the output of that layer's native MLP
   at the country-token position only. No `b_dec`, no encoder term, no subtraction of current activations, native MLP
   and attention untouched, everything downstream recomputed (no KV cache). beta in {0, 0.5, 1, 2} and nothing larger.
   A new class in a new file does this; dir31's `common.Edit` (donor replacement) is not used, and neither is the old
   layer-16 edit. Checks: beta = 0 and hook removal must reproduce the unedited logits.
4. Answer effect = change in logit(` Berlin`) - logit(` Tokyo`) at the last position, after the fixed bridge ` is`,
   edited minus unedited. "+1 logit toward Berlin" means this difference rises by at least 1.0. The change in
   log P(Berlin) is saved next to it. [Raw logit(Berlin) alone: a uniform logit shift would count as an effect.]
5. Three preservation checks, exactly the feedback's tolerances, comparisons inclusive, no rounding:
   - Readout: |readout_edited - readout_unedited| at layer 26, country token of the capital prefix, at most 10% of
     (unedited Japan readout - unedited Germany readout) for the same template. Two-sided. Uses dir31's frozen
     `results/probe.npz` unchanged. For a layer-26 edit the readout sees the added vector directly; that is noted, not
     corrected.
   - Controls: the 4 mid-sentence Japanese prompts from dir30 that Qwen answers correctly ("The official currency of
     Japan is the", "What is the official currency of Japan? It is called the", "In Japan, the primary language is",
     "People in Japan speak"). The same vector is added at the ` Japan` token of each, same layer. Each must keep its
     unedited top-1 (asserted equal to the correct answer token) and lose at most 0.05 of that answer's probability.
     Excluded: the 3 prompts that start with the country word (dir31 found the edit site and readout invalid there).
     German-language controls are not used; the feedback asks for Japanese ones.
   - Bridge: ` is` stays the top-1 next token at the country token of the capital prefix and its probability falls by
     at most 0.05.
   A probability that rises never fails a check. The layer-26 readout at the control prompts is saved as a diagnostic
   only; the feedback ties the readout tolerance to the capital template.
6. Screen on the tuning prompt only. Every candidate alone at beta = 0, 0.5, 1, 2, ranked by measured answer effect.
7. Groups. For each layer and each beta > 0: take the features that pass all three checks at that beta, rank them by
   measured answer effect at that beta, and combine the top 3 and the top 5 where that many passing features exist.
   The summed vector is applied at that beta and every check is rerun. A column flags groups whose members all had a
   positive single-feature effect. [Orchestrator default "passing AND positive effect": narrower than the feedback's
   wording "top 3 and top 5 preservation-passing features"; the literal version contains every group of the narrower
   one, and the flag marks them. Membership fixed at beta = 1 and reused at other strengths: passing is
   strength-dependent, so that would mix failing features into groups.] No other combinations, no cross-layer groups.
8. Random controls = dir31's construction: same layer and position, same number of features and the same
   coefficients, placed on random decoder rows of the same dictionary (dir31's stored 64 random rows per layer, seeds
   1-3), rescaled to the real vector's L2 norm.

Selection rule (fixed now, applied to tuning results only)
- A setting = (layer, feature list, beta > 0), a single feature or a top-3 / top-5 group.
- Qualifying = passes all three checks AND answer effect >= +1.0 on the tuning prompt. [Any positive effect: "smallest"
  would then freeze a near no-op. Berlin top-1: the feedback says Berlin need not become top-1.]
- Smallest = fewest features, then smaller beta, then larger answer effect.
- The chosen edit (layer, features, amplitudes, beta, vector L2 norm, native MLP-output norm at the site) is written to
  `results/additive/frozen_edit.json` before any edited run on a reserved prompt, and never changed afterwards.
- Pilot passes only if, on BOTH reserved prompts ("The capital of Japan is", "The capital of Japan is the city of"):
  answer effect >= +1.0; answer effect larger than each of the three random controls on that prompt; all three checks
  pass (readout tolerance from the "The capital of {}" template, same four controls, ` is` at the reserved prefix).
  Greedy completions: 8 new tokens, no KV cache.
- S4.4 runs only if the pilot passes: the same fixed vector at every one of the 41 path points, with baseline and the
  three random controls. Country preservation along the path = |readout_edited(t) - readout_unedited(t)| at layer 26
  within the same 10% tolerance at all 41 points, both endpoints included; otherwise the plateau result is labelled
  confounded. Hidden states are read at the country token in layers after the edited one (headline layer 26, or 27 if
  the edit is at 26). Saved for every condition: d(t) with its own endpoints, d(t) with the fixed unedited endpoints,
  raw step sizes, steps divided by that condition's mean step, endpoint separation.
- If nothing qualifies on tuning: no edited run on reserved prompts, no S4.4. Report the best answer effect overall
  with its country cost (all three checks), the best preservation-passing effect, and the feedback's sentence "No
  selective additive intervention found in this bounded candidate search." No strength above 2, no wider search.

Plan and checks
1. S4.1 candidates + amplitudes -> verify: 69 candidates, amplitudes match dir30's stored activations.
2. S4.2 screen -> verify: beta = 0 reproduces the unedited logits; every row carries all three check results.
3. Freeze from the screen CSVs alone -> verify: `frozen_edit.json` exists before the first reserved-prompt run.
4. S4.3 / S4.4 only as gated above. Re-hash the 57 protected files against the manifest at the end.

### 2026-10-11 — iteration 2 (end of the experiment phase): S4.1-S4.2 run, gate not met, reserved prompts and S4.4 not run

What I did
- `experiments/additive_common.py`: `AdditiveEdit` (adds one fixed vector to a layer's native MLP output at one
  position), the three preservation checks, dir31's matched random control. `additive_s1_candidates.py`: candidates
  and reference amplitudes (21 dictionaries streamed through /tmp, about 2 minutes). `additive_s2_screen.py`: the
  tuning-prompt screen. `additive_s3_freeze.py`: applies the rule declared above to the screen CSVs, no model run.
  `additive_plots.py`: four figures. `run_additive.sh`: one rerun command.
- Outputs under `results/additive/` (candidate_counts.json, candidates.csv/.npz, discovery_prompts.csv, site_norms.csv,
  tuning_base.json, screen_single.csv, screen_group.csv, screen_controls.csv, tuning_best_controls.csv,
  tuning_summary.json, frozen_edit.json, three logs) and `plots/additive/figA1-figA4`.

What I learned
- S4.1. 69 candidates in layers 3-26 (70 with Germany top10, 69 with top20). Re-encoded activations match dir30's
  stored ones to 5e-5. Amplitudes run from 0.14 (layer 3) to 42.5 (layer 25). Support: 18 features are positive on 7/7
  discovery prompts, 7 on only 2/7; all on both facts. The filter is loose in two ways the reader should know: 18 of the
  69 are active on at least one of dir30's Germany discovery facts (5 on all ten) and 26 on some Japan facts; and only
  14 have a dir30 discovery score >= 0.5, all in layers 23-26 (median score 0.17). Five candidates are already active
  at the Japan token of the tuning prompt.
- Checks of the machinery. beta = 0 and hook removal: max logit difference 0.0. With an edit at layer L the layer-L
  block output at the Japan token moves by the vector (max error 1.9e-5), earlier layers do not move, and the ` is`
  token is untouched up to layer L. The screen is deterministic (rerun gave identical CSVs).
- Unedited tuning prompt: logit(Berlin) - logit(Tokyo) = -9.526, P(Tokyo) 0.1835, P(Berlin) 1.3e-5, P(` is`) 0.409,
  readout +1.062, Japan-Germany gap 2.266, so the readout tolerance is 0.227.
- S4.2. 256 settings with beta > 0: 207 single-feature (69 x 3) and 49 groups (12 layers have at least 3 candidates).
  Answer effects range from -0.286 to +0.314. None reaches +1; 16 reach +0.1.
- Preservation is almost never the binding constraint: 250 of 256 settings pass all three checks. All 256 pass the
  readout check (largest change 8.5% of the gap), 253 pass the controls (largest fall 0.061; the correct top-1 is kept
  in every control run), 252 pass the ` is` check (largest fall 0.110, the one case where the next token becomes `,`).
- Best answer effect overall, which is also the best preservation-passing one: layer 17, top-3 group
  (15678 + 70371 + 150354), beta = 2: +0.314 (margin -9.526 -> -9.213, P(Berlin) 1.34e-5 -> 1.69e-5, P(Tokyo) 0.183 ->
  0.170). Its country cost: readout -0.041 (1.8% of the gap), largest control fall 0.008, ` is` fall 0.014. Vector L2
  25.1 against 30.0 for the native MLP output there. The same group gives +0.084 / +0.165 at beta 0.5 / 1.
- Best single feature: layer 17, 15678, beta = 2: +0.278, passes. Best setting that fails a check: layer 10, 48590,
  beta = 2: +0.208 with control fall 0.061 and ` is` fall 0.066.
- The 14 candidates with dir30 score >= 0.5 do nearly nothing here: best +0.046 (fails the ` is` check), best passing
  +0.021. The larger effects come from low-score features that dir30 also finds active on most city-answer prompts.
- Three matched random controls for the best setting on the tuning prompt: +0.052, +0.223, -0.126. The best effect is
  larger than each, by 0.09 at the closest; it sits in the range random directions of that norm produce.
- In 5 settings the top token after the bridge changes from ` Tokyo` to ` a` (P(Tokyo) 0.12-0.14). There the effect
  comes from Tokyo falling; log P(Berlin) falls slightly too (-0.059 for the layer-10 top-3 group at beta = 1).
- Gate. 0 qualifying settings. `frozen_edit.json` records gate = none_qualified. Per the rule declared before the
  screen: no edited run on a reserved prompt, no greedy completions, no S4.4. There is no plateau result to label.
  Conclusion in the feedback's words: "No selective additive intervention found in this bounded candidate search."

Decisions, additions and caveats
- Groups follow the feedback's wording (top 3 / top 5 of the preservation-passing features, any sign). 15 of the 49
  groups have only positive members; the best of those is layer 10 (137591 + 5895 + 1418), beta = 2, +0.305.
- Two things were added beyond the request, both descriptive and neither used for selection: the site check above,
  and the three random controls for the best setting on the tuning prompt.
- "Qualifying" needed a threshold the feedback does not spell out. I used the pilot's own +1 and kept it after seeing
  the screen. Under a looser reading (any positive, preservation-passing effect) the smallest edit would be layer 17,
  feature 15678, beta = 0.5 (+0.072 on tuning) and the reserved prompts would have been run. Flagged to the
  orchestrator; not run.
- No code was written for S4.3 evaluation or S4.4, because their gate was not met.
- Limits of this search: one tuning prompt; amplitudes from 7 prompts of 2 facts, read at a last-token position
  (` is` / ` of`) and applied at a country token; strengths capped at 2; within-layer groups only. This says nothing
  about whether a selective additive edit exists outside these bounds.

Revised next step: orchestrator audit of code and numbers, then S4.5 (addendum sections in RESULTS.md and REPORT.md,
CHANGELOG entry, manifest completion fields).

On track? yes — S4 experiments finished (S4.1-S4.2 run; S4.3 evaluation and S4.4 closed by the gate), about 70% of S4 done; S4.5 deliverables wait for the orchestrator's audit, no blocker.

### 2026-10-11 — iteration 2 (S4.5): deliverables written after the orchestrator's audit

Audit outcome passed to me by the orchestrator (standing in for `run.sh`): it recounted the screen and confirmed the
numbers; the gate stays as declared (all three tolerances and at least +1 on the tuning prompt); the sign-agnostic
groups stay; the tuning-prompt random controls stay in RESULTS.md as descriptive; one descriptive addition was
requested, a tuning-prompt completion for the best setting.

What I did
- New `experiments/additive_s4_tuning_example.py`, added to `run_additive.sh`: unedited model against the setting
  with the largest effect, tuning prompt only -> `results/additive/tuning_example.csv`. The greedy continuation is
  unchanged (` Tokyo. The capital city of China is`); P(Tokyo) 0.1835 -> 0.1696, P(Berlin) 1.34e-5 -> 1.69e-5,
  readout +1.062 -> +1.021. No existing results file was rewritten. A second end-to-end `bash run_additive.sh`
  reproduced all 13 CSV/JSON/NPZ outputs byte for byte and left `frozen_edit.json` untouched.
- RESULTS.md: Section 5 added (5.1-5.10, Figures 4-7, one equation, table of all 69 candidates). Opening lines,
  Headline, a note under Terms and the Files list extended. Sections 1-4 are byte-identical.
- REPORT.md: one section "Addendum: can an additive edit promote Berlin while preserving Japan?" before the
  Conclusion (Figure 4, one equation), one pointer paragraph in the Summary and one in the Conclusion, and the
  closing line names both rerun commands. 4,177 -> 4,990 words (limit 5,000), 3 -> 4 figures.
- CHANGELOG.md appended (plus a one-line correction of the word count). PLAN.md: S4 header and S4.5 ticked, status
  and next step updated.
- Manifest: all 41 items filled and set to done, the four embedded PNGs appended to `required_outputs`, state set
  to review_pending.

Checks, all run from this directory
- `workflow.py validate`: OK, also with the review-mode checks. `check_md.py`: OK for both files.
  `check_render.py`: ALL CHECKS PASS (REPORT.md 8 display equations and 4 figures; RESULTS.md 1 equation and 7
  figures). `workflow.py check-budgets`: OK. Caption lines: 11 for 11 embeds. The 57 protected files are unchanged
  against `baseline_sha256`.

Decisions and caveats
- Wherever a reader would expect reserved-prompt numbers, completions or plateau plots, both deliverables say that
  the request does not define "qualifying", that the definition was fixed before the screen, and that parts 3 and 4
  of the request produced no outputs. This is written as a condition not met on the tuning prompt. It is never
  described as a pilot that failed on reserved prompts.
- The REPORT addendum is tight: 10 words of budget remain, so detail lives in RESULTS.md Section 5.
- The candidates are never called Berlin or city features, and nothing says that answer and country cannot be
  separated.
- The evidence fields cite files under `results/additive/` that are not in `required_outputs`; the reviewer sees the
  two reports and the four PNGs.

Revised next step: none for the worker; the task waits for the wrapper's format checks and the content review.

On track? yes — S4 complete (S4.1-S4.5), 100% done and awaiting review; no blocker.

### 2026-10-11 — iteration 2 (review round 1): reserved-prompt pilot ordered by the operator; plan logged BEFORE any reserved-prompt forward pass

What happened
- The pre-registered rule above (qualifying = all three tolerances AND at least +1 on the tuning prompt) selected
  nothing: 0 of 256 settings. Under it I froze nothing and ran no reserved prompt.
- The independent reviewer rejected round 1. Main failure: part 3 of the request (freeze and evaluate on the reserved
  prompts) states no condition, its +1 bar is written for the reserved prompts, and my definition moved that bar to
  the tuning prompt, so a requested evaluation did not exist. Four smaller failures: a random-control sentence in
  REPORT.md without its caveat; a "largest effect" column that lists only the most positive value; a control prompt
  that RESULTS.md never names; "greedy continuation" undefined, dir30's cutoffs unexplained, figure numbers that
  differ between the two files.
- Operator ruling, relayed to me by the orchestrator session (I did not receive it directly): "qualifying" is
  resolved as the preservation-passing setting with the largest tuning answer effect, because none reached +1. That
  is layer 17, candidates 15678 + 70371 + 150354, beta 2 (+0.314 on tuning). It is run once on both reserved prompts
  against baseline and three matched random controls and reported as a pilot that is not expected to reach +1. The
  operator knows this is the largest passing edit, not the "smallest".

What this changes, stated plainly
- This freeze rule was set AFTER the tuning screen and BEFORE any reserved-prompt run. For part 3 of the request it
  replaces my journaled +1-on-tuning gate. The outcome of that gate (nothing reached +1 on tuning) stays on record
  and is reported.
- Up to this entry no reserved prompt has been run under any edit, and this task has made no unedited reserved-prompt
  pass either (unedited values exist from dir31's S1).

Freeze (mechanical, from the tuning CSVs only)
- Rule: among all settings with beta > 0 in screen_single.csv and screen_group.csv that pass all three tolerances,
  take the one with the largest answer_effect. I expect the layer-17 top-3 group at beta 2 and check it afterwards;
  the script does not hard-code it.
- `additive_s3_freeze.py` now writes two files. `preregistered_gate.json` holds the outcome of the +1-on-tuning rule
  (none qualified). `frozen_edit.json` holds the edit frozen under the ruling: layer, features, amplitudes, beta,
  vector L2, the native MLP-output L2 at the site for the tuning prefix and (from dir31's existing s2_slices.csv)
  for the reserved prefix, and the statement that nothing met the +1 tuning bar. The old frozen_edit.json said
  `gate: none_qualified, reserved_prompts_run: false`. It is replaced once, because the second field is about to
  become untrue; its remaining content moves to preregistered_gate.json. After that the never-rewrite check
  protects the new file.

Reserved evaluation plan (one pass; no retuning whatever it shows)
- Prompts: "The capital of Japan is" and "The capital of Japan is the city of". The edit goes at the ` Japan` token
  (position 3), layer and vector exactly as frozen.
- Arms: unedited; frozen edit; three random controls (same layer and position, same feature count and coefficients
  on dir31's stored random decoder rows, rescaled to the frozen vector's L2 norm, seeds 1-3: the construction and
  seeds already used on the tuning prompt).
- Per arm and prompt: change of logit(Berlin) - logit(Tokyo) after the bridge; P(Tokyo), P(Berlin), top-1; layer-26
  readout change against 10% of the unedited Japan-Germany gap of "The capital of {}"; the ` is` check at that
  prefix's country token; the four Japanese controls with the arm's vector at their ` Japan` token; a greedy
  completion (8 new tokens, no KV cache, hook active on every pass).
- Verdict: three criteria, reported separately and each on both prompts. (1) Answer effect >= +1.0. (2) Answer
  effect larger than each of the three controls on that prompt. (3) All three preservation checks pass for the
  frozen edit. The pilot passes only if all three hold on both prompts. Tolerances are inclusive; "larger than" is
  strict.
- Part 4 stays conditional: the 41-point path runs only if the pilot passes all three, with the path rule journaled
  above. Otherwise it is not run and the write-up gives the pilot result as the reason.
- Outputs: new script `additive_s5_reserved_eval.py`; `results/additive/reserved_eval.csv`,
  `reserved_controls.csv`, `pilot_verdict.json`, a log; one figure under `plots/additive/`.
- The stop-rule sentence stays if the pilot does not pass.

Review failures 2-5 are repaired in the same round; details follow in the closing entry.

### 2026-10-12 — iteration 2 (review round 1, end): pilot run on the reserved prompts, findings 2-5 repaired, resubmitted

Timeline (pod clock, UTC). 23:53 plan entry above appended. 23:54:45 `frozen_edit.json` written (sha256
846998fb...). 23:55:32-23:55:50 reserved-prompt pilot, one pass. 00:02-00:05 full `bash run_additive.sh`: all 17
CSV/JSON/NPZ outputs reproduced byte for byte, `frozen_edit.json` untouched.

Frozen edit (derived mechanically from the screen CSVs): layer 17, candidates 15678 + 70371 + 150354 (amplitudes
11.894, 2.155, 1.658), beta 2, vector L2 25.12; native MLP-output L2 30.04 at the tuning prefix and 29.92 at the
reserved prefix (the live value in the pilot equals dir31's stored one). The old `frozen_edit.json` (gate
none_qualified, reserved_prompts_run false) was moved out of results/ and replaced once; the outcome of the
pre-registered gate now lives in `preregistered_gate.json`.

Pilot result (nothing retuned afterwards)
- "The capital of Japan is": frozen edit +0.381 (logit difference -10.145 -> -9.764; P(Tokyo) 0.402 -> 0.357;
  P(Berlin) 1.58e-5 -> 2.05e-5); random controls +0.201, +0.160, -0.085.
- "The capital of Japan is the city of": frozen edit +0.293 (-11.575 -> -11.282; P(Tokyo) 0.356 -> 0.346;
  P(Berlin) 3.3e-6 -> 4.4e-6); random controls +0.197, +0.146, -0.022.
- Preservation at the reserved prefix: readout -0.026 (1.2% of the 2.261 gap) passes; controls pass (largest fall
  0.008); ` is` fails: P 0.693 -> 0.598, a fall of 0.095, still top-1. The random controls lower P(` is`) by
  0.033-0.043; control 1 fails the control check (falls 0.060 and 0.053).
- Greedy completions under the frozen edit are unchanged (` Tokyo. The capital of the United States` on both
  prompts). Random control 3 gives ` Tokyo. The capital of Japan is also`.
- Verdict: criterion 1 (+1) not met; criterion 2 (above every control) met; criterion 3 (preservation) not met.
  The pilot does not pass, S4.4 was not run, and the conclusion sentence stays.

What I learned
- Passing the checks on the tuning prefix did not transfer: the same vector lowers P(` is`) by 0.014 on "The
  capital city of Japan" and by 0.095 on "The capital of Japan".
- The answer effect on the reserved prompts (0.29-0.38) is the size seen on tuning (0.31). It is above the three
  controls by 0.10-0.18 logits, which is weak evidence with three seeds.

Repairs for review findings 2-5
- 2: the tuning-prompt random-control sentence is gone from REPORT.md. RESULTS.md keeps the comparison in 5.9 as
  reference values, with the caveat (a random direction reaches +0.223; selection as the maximum; three seeds).
- 3: the per-layer table has "highest" and "lowest" single-candidate effect columns; a new paragraph reports the
  moves away from Berlin (five settings below -0.2, 122 of 256 negative). Checking the same framing elsewhere
  turned up one more case: the paragraph on high-scoring dir30 candidates said they "do least", but one of them
  has -0.211. It now gives the most positive and the most negative value. "Largest effect" wording is
  sign-explicit in both files.
- 4: the 0.061 fall is on "What is the official currency of Japan? It is called the" (P(yen) 0.383 -> 0.321),
  read from screen_controls.csv. RESULTS.md names the control with the largest fall for every failing setting.
- 5: "greedy continuation" is defined under Terms in RESULTS.md and inline in REPORT.md; dir30's cutoffs (10, 20,
  50 best-scoring features per layer) are explained; the screen figure is Figure 4 in both files.

Files
- New: `experiments/additive_s5_reserved_eval.py`; `results/additive/preregistered_gate.json`,
  `reserved_eval.csv`, `reserved_controls.csv`, `pilot_verdict.json`, `s5_reserved_eval.log`;
  `plots/additive/figA5_reserved_pilot.png`.
- Changed: `additive_s3_freeze.py` (two-step rule), `additive_plots.py` (figA5 replaces figA4; one label in
  figA3), `run_additive.sh`, `results/additive/frozen_edit.json` (replaced once, see above), RESULTS.md Section 5,
  REPORT.md addendum, CHANGELOG.md, PLAN.md, the manifest. Removed: `plots/additive/figA4_best_vs_random.png`.

Checks, all from this directory: `workflow.py validate` OK (also in review mode); `check_md.py` OK for both files;
`check_render.py` ALL CHECKS PASS (REPORT.md 8 equations, 4 figures; RESULTS.md 1 equation, 7 figures);
`check-budgets` OK (REPORT.md 4,988 of 5,000 words); 11 caption lines for 11 embeds; the 57 protected files are
unchanged against `baseline_sha256`. Manifest: 41 items done, `operator_rulings` added, state review_pending.

Decisions and caveats
- The freeze rule used for the pilot was set after the tuning screen, on the operator's ruling as relayed by the
  orchestrator. Both deliverables state this as part of the method, together with "largest passing, not
  smallest".
- The tuning-only bar figure (figA4) was dropped; the left panel of figA5 shows the same four values.
- REPORT.md has 12 words of budget left.

Revised next step: none for the worker; the task waits for the wrapper's format checks and the content review.

On track? yes — S4 complete after review round 1 (pilot run, does not pass; S4.4 not run), 100% done and awaiting review; no blocker.

### 2026-10-12 — iteration 2 (review round 2): wording and disclosure repairs, resubmitted

The round-2 reviewer accepted the substance and rejected on four points. The orchestrator's own audit agreed with
the round-1 numbers. Nothing was rerun on the model in this round: the pilot stays as run, one pass.

What I did
- Terminology. "Controls" meant both the matched random edits and the four Japanese prompts, in adjacent
  sentences. Both files now say "random control(s)" and "control prompt(s)" throughout the addendum; table headers
  and "Checks failed" cells follow. `additive_plots.py`: the figA5 annotation reads "fails control prompts".
  Regenerating left figA1, figA2 and figA3 byte-identical, and no results file changed.
- Random-control pool. I had described the controls as random directions of the layer-17 dictionary. They are
  picked from dir31's stored pool of 64 random decoder rows for that layer (`results/slices/layer_17.pt`), three
  per seed, and two rows repeat: 152238 (second amplitude in random control 1, largest in random control 2) and
  74060 (smallest amplitude in random controls 2 and 3). Checked without a model run: the pool has 64 rows, the
  three picks reproduce the rows stored in the CSVs, and no pool row is in a dir30 feature set. RESULTS.md 5.1,
  5.7 and 5.10 now state this; REPORT.md reports criterion (b) as "met, weakly" with the pool and the sharing.
- REPORT.md: the Question paragraph introduces the operator's follow-up request and says the operator fixed the
  freezing rule; the plateau-test sentence cites RESULTS.md Sections 5.7 and 5.8.
- Offsetting trims inside the addendum only. REPORT.md 4,988 -> 4,992 words; 4 figures.
- CHANGELOG appended; PLAN status and next step updated; manifest items for the freeze and for criterion 2 now
  mention the pool, `review` record kept, state set to review_pending.

What I learned
- Reusing dir31's control construction carried a property I had not disclosed in the first phase: a pool of 64
  rows makes repeated rows likely with three seeds. It was in my first report to the orchestrator as an
  implementation note but not in the deliverables. A fresh draw from all 163,840 rows would have avoided it; the
  operator-ordered pilot is not to be redrawn, so the limitation is stated instead.

Checks, all from this directory: `workflow.py validate` OK (also in review mode); `check_md.py` OK for both files;
`check_render.py` ALL CHECKS PASS (REPORT.md 8 equations, 4 figures; RESULTS.md 1 equation, 7 figures);
`check-budgets` OK (REPORT.md 4,992 of 5,000 words); 11 caption lines for 11 embeds; the 57 protected files are
unchanged against `baseline_sha256`; `frozen_edit.json` hash unchanged (846998fb...).

Revised next step: none for the worker; the task waits for the wrapper's format checks and the content review.

On track? yes — S4 complete after review round 2 (wording and disclosure only), 100% done and awaiting review; no blocker.

### 2026-10-13 — iteration 3: finalize only, STOP written

No unaddressed feedback: `human_feedback_additive_berlin.md.addressed.md` is the only feedback file and its
manifest state is `addressed`. The success criterion is met (S1-S3 "inseparable" result; S4 approved), so no
experiment was run and no deliverable was edited. PLAN.md said STOP was restored, but the file was absent on
disk; I wrote it.

Checks: `check_render.py` ALL CHECKS PASS (REPORT.md 8 equations, 4 figures; RESULTS.md 1 equation, 7 figures);
11 caption lines for 11 embeds.

On track? yes — plan complete, 100% done, STOP present; no blocker.
