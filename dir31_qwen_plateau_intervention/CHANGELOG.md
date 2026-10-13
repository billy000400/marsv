# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-10-09 — first results

- RESULTS.md: replaced the template with the full evidence record. Sections: baseline plateau and readout (S1), scan,
  frozen edit, capital and context prompts under the edit (S2), path under the edit (S3), verdict against the plan's
  table. Three figures embedded (plots/fig1_baseline_plateau.png, fig2_intervention.png, fig3_path_comparison.png).
- REPORT.md: replaced the template with the synthesis. Core question: does the plateau at the country token follow
  country identity or the later capital answer. Verdict: not disentangled. Three figures, three tables, 7 display
  equations. No earlier numbers existed, so nothing was superseded.
- Framing note: the plan hoped for an answer-specific early edit. The evidence gave a country swap, so the report is
  built around the inseparable result and the two things the interventions do establish (layer-16 country features
  drive capital, currency and language answers at beta = 2; the late-layer plateau depends on the same features).

## 2026-10-11 — addendum: additive Berlin edit (operator feedback `human_feedback_additive_berlin.md`)

- RESULTS.md: added Section 5, a separate addendum. It covers the candidates and reference amplitudes, the unedited
  tuning prompt and control prompts, checks of the edit, the 256-setting screen with all three preservation checks,
  the best and the failing settings, the groups, a descriptive random-control comparison and a tuning-prompt example,
  what was not run and why, the limits, and a table of all 69 candidates. Four new figures (Figures 4-7,
  `plots/additive/figA1`-`figA4`) and one display equation. The opening lines, the Headline, a note under Terms and
  the Files list were extended to cover Section 5. Sections 1-4, their tables and Figures 1-3 are unchanged.
- REPORT.md: added one section, "Addendum: can an additive edit promote Berlin while preserving Japan?", placed
  before the Conclusion, with one figure (Figure 4) and one display equation. One short pointer paragraph was added
  to the Summary and one to the Conclusion, and the closing line now names both rerun commands. 4,177 -> 4,986
  words by the gate's counter, 3 -> 4 figures. No existing finding was changed.
- Result recorded: no setting qualified on the tuning prompt. The largest answer effect is +0.314 logits against the
  +1.0 required, and 250 of 256 settings pass all three preservation checks. Nothing was frozen; the reserved
  prompts and the path were not run. "No selective additive intervention found in this bounded candidate search."
- Nothing was superseded. The addendum is a new experiment with a different edit; the S1-S3 numbers stand.
- Framing note: the report's verdict ("not disentangled") is unchanged. The addendum adds a second, independent
  search that did not find the edit the test needs. Neither deliverable claims that no such edit exists.
- Correction to the entry above: after a final wording pass REPORT.md has 4,990 words (the entry says 4,986); figures unchanged at 4.

## 2026-10-11 — review round 1: reserved-prompt pilot added, four reviewer findings repaired

- Why: the independent reviewer rejected the first submission. Its main finding was that part 3 of the request
  (freeze and evaluate on the reserved prompts) had not been run, because "qualifying" had been defined as +1 on
  the tuning prompt and nothing met it. The operator then resolved "qualifying": freeze the preservation-passing
  setting with the largest tuning answer effect and run it once on the reserved prompts as a pilot.
- New result in both deliverables: the frozen edit (layer 17, candidates 15678 + 70371 + 150354, beta 2) on the two
  reserved prompts gives +0.381 and +0.293 logits toward Berlin, against +0.201 / +0.160 / -0.085 and +0.197 /
  +0.146 / -0.022 for the three matched random controls. Pilot criterion 1 (+1 logit) not met, criterion 2 (above
  every control) met, criterion 3 (preservation) not met: P(" is") at the reserved prefix falls by 0.095. The pilot
  does not pass, the plateau test stays not run, and the conclusion sentence is unchanged.
- Superseded statements (old -> new): "nothing was frozen; the reserved prompts were not run" -> "the largest
  passing setting was frozen and run once on the reserved prompts; the pilot does not pass". The definition of
  "qualifying" as +1 on the tuning prompt -> a two-step rule (that definition first, then the operator's
  resolution when nothing reaches +1). No number from the first submission changed.
- RESULTS.md Section 5: Outcome rewritten; 5.1 defines "Frozen edit" and "Pilot criteria" and now carries
  Figure 4 (the screen); new 5.6 (frozen edit), 5.7 (pilot: per-arm table, verdict table, control-prompt table,
  Figure 7) and 5.8 (plateau test not run). The tuning-prompt random controls and tuning-prompt example moved to
  5.9 as reference values; Limits is 5.10 and the candidate table 5.11. Headline, Terms and Files updated.
- RESULTS.md, reviewer findings: (a) the per-layer table now gives the highest and the lowest single-candidate
  effect, and a new paragraph reports the moves away from Berlin (down to -0.286); "largest effect" wording was
  made sign-explicit throughout; the paragraph on high-scoring candidates now gives their most negative effect
  (-0.211) too. (b) The control prompt behind each "largest control fall" is named (0.061 is on "What is the
  official currency of Japan? It is called the"). (c) "Greedy continuation" is defined under Terms; dir30's three
  per-layer cutoffs are explained. (d) Figure numbers: the screen figure is Figure 4 in both files; RESULTS.md
  Figures 5, 6, 7 are the candidates, the preservation scatter and the pilot. The tuning-only bar figure (figA4)
  was dropped; its content is the left panel of the new Figure 7 (figA5).
- REPORT.md addendum rewritten within the budget: paragraphs Question, Candidates, Additive edit, Tolerances,
  Screen and freeze, Result on the tuning prompt, Result on the reserved prompts, Conclusion. The tuning-prompt
  random-control sentence and the "What was not run" paragraph were removed; "greedy continuation" is defined;
  effects are given with both signs. Summary and Conclusion pointers updated. 4,990 -> 4,988 words, 4 figures.
  No pre-existing finding was changed.

## 2026-10-12 — review round 2: wording and disclosure repairs (no number changed, nothing rerun)

- Why: the reviewer accepted the substance (request carried out in full, numbers consistent) and rejected on four
  wording and disclosure points.
- "Controls" named two things. Both deliverables now write "random control(s)" for the matched random edits and
  "control prompt(s)" for the four Japanese currency and language prompts. RESULTS.md Section 5: the check is
  "Control-prompt check", table columns read "Largest control-prompt fall", "Checks failed" cells read "control
  prompts", and 5.1 states the convention. The annotation in Figure 7 (figA5) reads "fails control prompts"; the
  other three figures are byte-identical.
- Random-control pool disclosed. RESULTS.md 5.1 says the random directions come from a pool of 64 decoder rows per
  layer drawn once by the main study; 5.7 lists the rows of each random control and the two shared rows (152238 in
  random controls 1 and 2, 74060 in random controls 2 and 3); the "weak evidence" sentence and the 5.10 limits say
  the same. REPORT.md reports criterion (b) as "met, weakly" and says the three share directions from a pool of 64.
  The random controls were not redrawn and the reserved evaluation was not rerun.
- REPORT.md addendum, Question paragraph: one clause says the addendum answers a follow-up request from the
  project's operator, who also fixed the freezing rule.
- REPORT.md cross-reference corrected: the plateau test not being run now cites RESULTS.md Sections 5.7 and 5.8.
- Offsetting trims inside the addendum (wording only): Question, Screen and freeze, both Result paragraphs, the two
  pointer paragraphs. REPORT.md 4,988 -> 4,992 words, 4 figures. No result, table value or figure number changed.
