# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-10-09 — iteration 1: S0–S5 run end to end

**Did.** Read `../BUDGET.md` and `../CLAUDE.md` (no `AGENTS.md` exists). Downloaded pinned Qwen3-0.6B and the pinned LRE files. Built the four answer groups, controls and backgrounds; ran the correctness filter (S1); cached last-token MLP inputs and encoded them with all 28 per-layer transcoders (S2); computed sets, overlap (S3), the early country-token readout (S4) and the small suppression check (S5). Wrote `REPORT.md`, `RESULTS.md`.

**Learned.**
- Surviving discovery facts / prompts: Tokyo 6 / 18, Japan 16 / 21, Berlin 2 / 7, Germany 10 / 10. Backgrounds are thin (37 of 114 sampled facts correct).
- Top-20 sets: 276 / 403 / 85 / 271. Tokyo–Japan share 58 (Jaccard 0.093); Berlin–Germany share 14 (0.041). Japan–Germany is the highest pair (0.140), so pooled overlap tracks prompt format.
- Strong features (score >= 0.5): 16 / 20 Tokyo features are in the Japan set, 0 / 20 in the Germany set; Berlin 11 / 25 in Germany, 0 / 25 in Japan.
- Early: 7 / 276 Tokyo-set and 7 / 85 Berlin-set features are active on the country token; the country-specific ones are mostly the shared strong features.
- Suppression of one feature per city: null (P(city) moves by at most 0.013; controls by at most 0.021).

**Assumptions and decisions (logged because nobody can be asked).**
- Disk quota: the shared HF cache volume refused writes after three transcoder layers. Remaining layers are streamed one at a time through `/tmp/hf_dir30` and deleted after use. No other user's cache was touched.
- Model run in float32 on plain prompts with the tokenizer's defaults (Qwen adds no start token). S0 showed the transcoder fails on the first token of a prompt; prepending a start token did not help, so prompts stay plain. No analysed position is a first token, except four context-control prompts in S5, which are flagged.
- Correctness aliases declared before any feature result: leading "the"; Yen = "Japanese yen"; New York City = "New York"; United States = "USA" etc. Rejected alternative: accepting the answer anywhere in the completion (forbidden by the plan).
- Background is drawn only for the four relations that contain discovery facts. The three Kyoto control facts and facts whose answer is not a city in `company_hq` are excluded from background. Rejected alternative: also drawing background for `city_in_country`, which no group uses.
- Prompts where the answer string appears in the prompt are dropped as leakage (none in the primary groups).
- Within-country city control: explicit list = `company_hq` facts with answer Kyoto (the only non-Tokyo Japanese city present). Qwen answers none correctly.
- Bug found and fixed before finalising: features with a true score of 0 were selected through float32 rounding noise (71 features, mostly Japan). Scores are now float64 with a 1e-6 floor. Old to new numbers are in CHANGELOG.
- Added, outside the plan: (a) the split of overlap by score (threshold 0.5, with 0.25 and 0.75 as checks), because the pooled sets are dominated by weak features and the pooled Jaccard alone would mislead; it is labelled post hoc in both deliverables. (b) An extra suppression row for shared early-active features, labelled "extra"; the predeclared set-exclusive rows are reported first.
- S5 control for Berlin (layer 25) is poorly norm-matched (0.75 vs 10.95) because only three eligible features exist at that token; reported as such.

**Next step.** None required: the success criterion is met. If reopened, the most useful additions would be more model-known Berlin facts and a multi-feature, multi-layer suppression.

On track? yes — all stages S0–S5 complete, 100%, no blocker.
