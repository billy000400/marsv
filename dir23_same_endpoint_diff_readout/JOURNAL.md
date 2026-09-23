# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-08-12 — S1–S5 complete in one iteration; verdict = aligned transitions

**Did.** Built `experiments/{common,s1_endpoints,s2_interp,s3_plots}.py`. S1: verified all
tokenizations and reproduced every value in the PLAN's preliminary endpoint table to the quoted
precision (immediate JSD 0.00761 bits; readout JSDs 0.991 / 0.885 / 0.915 / 0.968 / 0.111 bits, all
top-1 answers as expected), so no discrepancy-and-stop was triggered. S2: one shortest-arc slerp
interpolation (norm interpolated linearly) of the ` Japan`→` Germany` input embedding, 101 points,
inserted before layer 0, reused unchanged across all five readout suffixes. S3/S4: eight figures and
the transition table.

**Learned.** t50 = 0.454 / 0.444 / 0.443 / 0.450 for Capital / Continent / Currency / Language
(Type control 0.438). Δt50 = 0.011 — one grid step, ~1/25 of the transition width — so the four
readouts are descriptively aligned. All curves monotonic, single crossing at each threshold, widths
0.255–0.279 vs 0.80 for a linear change, so every readout plateaus and then switches. Top-1 flips
at t = 0.44–0.47. The immediate next-token prediction never changes (newline top-1 at all 101 points,
p 0.929→0.945), confirming the country information is only visible through a later readout.

**Assumptions / decisions logged (no human to ask).**
- Used the locally cached `gpt2-large` checkpoint; it is the same model as
  `openai-community/gpt2-large` (the org prefix is an alias) and avoids a download.
- Did not persist the raw 5 x 101 x 50257 logit matrices (51 MB). d(t) is computed from the full
  logit vectors in memory as specified; the saved machine-readable artefacts are the derived curves
  (`interp.csv`, `interp.npz`) plus `transitions.json`. Rejected alternative: commit the fp16 dump —
  rejected as repo bloat when `s2_interp.py` regenerates it in about a minute.
- Recorded the immediate position's d(t) as well (t50 = 0.438, w = 0.466). Reported in RESULTS.md
  only, with the caveat that d is normalized and this curve describes a 0.0076-bit change — it is
  not evidence of a fifth aligned transition.
- Deliberately added no analysis beyond the PLAN (no layerwise, probe, or new score work).

**Report scope.** REPORT.md embeds 7 figures (immediate prediction, five individual d(t), transition
comparison) at ~2.9k words, inside the 8-figure / 5000-word limit. The overlay figure lives in
RESULTS.md only.

**Check status.** `check_render.py`'s local half passes on both files (KaTeX compiles every display
and inline equation after GitHub backslash-stripping, no denylisted macros, every embed captioned,
every table motivated, contrast constructions within budget). Its GitHub markdown API call returned
HTTP 403 rate-limit on every retry over ~10 minutes (shared host IP), so equation *placement* was
verified with a local audit instead: all 5 display equations are ` ```math ` fences at column 0 with
blank lines either side and none inside a list item. Worth re-running the API half on a later
iteration if the direction is reopened.

**Next step.** None — success criterion met; STOP written.

On track? yes — S5, 100% done, no blocker.

## 2026-09-23 — human_feedback_0.txt: top-1 token added to Figure 7

**Did.** Used the per-t top-1 ids and answer probabilities already stored in `results/interp.npz`
(no model re-run). Rewrote the Figure 7 block of `s3_plots.py`: two panels (full sweep + 0.40–0.50
zoom), a strip of 101 open/filled squares per readout for the top-1 token, dashed line at the observed
switch. Wrote `results/top1_tokens.csv`. Updated REPORT.md (Methods metric, Figure 7 text, Table 2,
Summary, Conclusion) and RESULTS.md.

**Learned.** Top-1 is always a_A or a_B in each primary readout, switching once. t_switch
(interpolated zero crossing of p_A − p_B) = 0.460 / 0.432 / 0.461 / 0.447 for Capital / Continent /
Currency / Language against t50 0.454 / 0.444 / 0.443 / 0.450. They do not coincide exactly: Currency's
answer switches 0.018 after t50 (d = 0.61 at the switch; yen 0.22 vs euro 0.20 at t = 0.46, so the
argmax is fragile); Continent switches 0.011 before t50. Switch spread 0.029, still < 0.05.

**Decisions.** "Switch location" = linear-interpolated crossing of p_A − p_B, which matches how t50 is
interpolated; the bracketing grid points are shown too, since the switch is observed only on the grid.
Rejected: reporting only the first grid point with the new token (earlier practice), because it is
biased late by up to one step and cannot be compared fairly with an interpolated t50.

**Gotcha.** `s3_plots.py` regenerates every plot, and matplotlib PNG bytes are not reproducible, so
the seven must-remain-unchanged PNGs were restored from HEAD after running it (hashes verified equal
to the manifest baseline).
