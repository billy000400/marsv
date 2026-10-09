# PLAN â Qwen3-0.6B: Tokyo/Japan and Berlin/Germany feature overlap

> Suggested working folder: `qwen_city_country_features` (use an existing assigned folder if present).
> Agent REWRITES "Current status" / "Next step" and ticks stages each iteration.
> Disk (`PLAN.md`, `JOURNAL.md`, `RESULTS.md`, `REPORT.md`, `CHANGELOG.md`, plus `../BUDGET.md` and `../CLAUDE.md`) is the working memory.

## Question and scope

Find candidate transcoder features associated with **Tokyo**, **Japan**, **Berlin**, and **Germany**, then measure how many feature IDs are shared. Include Germany/Berlin as a parallel comparison using exactly the same procedure. `Germany` is the country; `German` is a language answer used only in a separate context check.

The required first result is a clear answer to:

1. After Qwen correctness filtering, how many distinct facts and prompts remain for each entity?
2. How many features are selected for each entity, and how many overlap within Tokyo/Japan and Berlin/Germany?
3. Do the selected features activate at the country position **before `is`** in the held-out capital prompts?

A small suppression check follows only after these outputs exist and usable early features are found. No claim that a candidate is a pure entity representation, or that overlap establishes planning.

## Success criterion (definition of done)

- `REPORT.md` contains the sample-count table, feature-count/overlap table, feature examples, and early-activation check, with concise Methods and figures.
- Machine-readable retained/rejected samples, feature rankings, selected feature IDs, overlap counts, and exact run configuration are saved.
- Optional suppression results, or a specific reason they were not run, are reported separately.
- Empty sets, scarce model-known examples, absent early activation, and null intervention results are valid completed findings. Never invent samples or silently loosen rules to obtain a positive result.
- When complete, write an empty `STOP` file. If execution is blocked, describe the blocker and incomplete stages explicitly; do not present setup checks as experimental results.

## Fallback (if time runs short)

Respect `../BUDGET.md`; reserve the last 20 minutes for results and `STOP`. Finish correctness filtering and the four count rows first. If a full layer scan will not fit, declare the fixed pilot layers `[4, 8, 12, 16, 20, 24]` **before inspecting feature results**, and use that same set for all groups. Label the output as a partial layer scan. Skip suppression before sacrificing the required counts and overlap analysis.

## Setup (fixed)

- Read `../BUDGET.md`, `../CLAUDE.md`, and applicable `AGENTS.md` before work and follow local scheduler rules. If instructions are absent, record that; do not invent GPU allocations or a runtime budget.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax.** Preserve the cluster CUDA environment. Prefer native Hugging Face inference and a per-layer transcoder loader; do not install the entire tracing stack if it would replace protected dependencies.
- Model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`. Use its matching tokenizer, plain text completion, `eval()`, no gradients, deterministic greedy decoding, and no chat template. Record tokenizer options, dtype, package versions, device, revisions, and seed `0`.
- Transcoders: `mwhanna/qwen3-0.6b-transcoders-lowl0`, revision `28aefe686c09e9bc1a862195b51e145f10868d87`. These are **per-layer transcoders (PLTs), not a CLT**, with files for zero-based layers `0..27`.
- Verified repository configuration: model name above; `feature_input_hook: mlp.hook_in`; `feature_output_hook: mlp.hook_out`. Training configuration reports post-normalization input (`before_ln: false`), ReLU activation, `d_model=1024`, `d_feature=163840`, no skip connection, and unit input/output scales. Check each loaded checkpoint's metadata and shapes rather than assuming the single training-config file describes every tensor correctly.
- For native HF Qwen, obtain `x` from the actual input to `model.layers[L].mlp` (after the block's post-attention RMSNorm), and `y` from that MLP's output. Do not feed an unnormalized residual stream to the encoder or normalize twice.
- Reference implementation: `decoderesearch/circuit-tracer`, inspected revision `0968d11a0e14a51d2fa4c445466949d2a93a5896`, `circuit_tracer/transcoder/single_layer_transcoder.py`. Its `load_transcoder` and `encode` are useful references; inspect the local version before using them. In that implementation `W_enc` has feature rows and decoder directions are rows of `W_dec`; verify orientation from shapes.
- Keep original Qwen MLPs active during screening. Load/encode one transcoder layer at a time, caching only the needed Qwen activations and sparse feature values. Do not require all 28 large dictionaries on the GPU together. Do not download visualization feature binaries or build attribution graphs.
- Before the main scan, check one layer: input/output dimensions match; encoding is finite and sparse; decoding reconstructs the expected output shape; record mean active-feature count and reconstruction relative L2 error on a small mixed batch. A large error is a reason to inspect hooks/scaling, not to tune entity-selection thresholds.
- `RESULTS.md` / `REPORT.md` contain current-best findings only; `CHANGELOG.md` records history.

## S1 â Build and filter the dataset

### Source and group definitions

Use `evandez/relations` (LRE), revision `1b9ec3cf2b8368e42bde7e80fcef384312a6ec07`, files under `data/factual/`:

- City-answer relations: `country_capital_city.json`, `country_largest_city.json`, `company_hq.json`.
- Country-answer relations: `city_in_country.json`, `landmark_in_country.json`, `food_from_country.json`.
- Separate context controls: `country_currency.json`, `country_language.json`.

Construct the **four primary answer groups** using exact source object labels:

| Group | Fact-selection rule | Raw facts | Raw unique prompts |
|---|---|---:|---:|
| Tokyo | object = `Tokyo` in city-answer relations | 13 | 52 |
| Japan | object = `Japan` in country-answer relations | 37 | 78 |
| Berlin | object = `Berlin` in city-answer relations | 10 | 40 |
| Germany | object = `Germany` in country-answer relations | 61 | 122 |

These counts were checked in the pinned data; **they are not model-filtered counts**. Recompute them on cluster. Japan comprises 35 `landmark_in_country` and 2 `food_from_country` facts; Germany has 61 `landmark_in_country` facts. Neither country occurs as an object in this snapshot's `city_in_country` file.

Here "Japan-related" operationally means **the expected answer is Japan**. It does not mean every prompt carrying any Japan information. Preserve that wording in the report.

Also count, evaluate, and report separately:

- Country-context controls: subject `Japan` or `Germany` in currency/language files, expected answer different from the city/country. Raw coverage: Japan 2 facts / 11 prompts; Germany 1 fact / 6 prompts. `Japan -> Japanese` and `Germany -> German` belong here, not in the primary country-answer groups. Germany currency is absent; do not silently add it.
- Within-country city controls already available in `company_hq`: non-Tokyo Japanese cities, e.g. Nintendo -> Kyoto. Keep the explicit inclusion list and provenance. Do not automatically infer company nationality from a name.
- Source-label anomalies: `company_hq` contains some country-valued objects (including Japan). Do not add these to the country-answer group: the template asks for a city. Record excluded anomalies.

For each relation, use the deduplicated union of `prompt_templates` and `prompt_templates_zs`. Preserve source text and format `{}` with the subject; never append the answer to feature-extraction input. Deduplicate `(relation, subject, object)` facts and identical `(prompt, answer)` rows. Flag conflicting labels, malformed rows, and literal answer leakage for review. Preserve raw files and their license notice.

### Holdout and background

- Reserve **all paraphrases** of `Japan -> Tokyo` and `Germany -> Berlin` in `country_capital_city` as held-out target facts. They count in overall model-filtered totals but never in feature-discovery totals or feature ranking. Their raw size is one fact / four prompts per city.
- For each relation used in discovery, choose up to 32 distinct background facts with seed `0`, excluding all four target answers and literal target-country subjects. Apply the same templates and correctness filter. Country-valued objects in city-answer relations remain excluded.
- Use background rows matched to the positive relation and template when scoring features. If a comparison cell has no correct background, mark it unavailable; do not substitute an unmatched cell silently.

### Qwen correctness filter

Run **unmodified Qwen** on every candidate/background prompt. Greedily generate at most 16 new tokens; save the raw completion and token IDs. A retained completion must begin with the expected answer after normalizing whitespace, case, and surrounding punctuation, with a word boundary after the answer. A predeclared leading article or explicit alias is allowed only if recorded in the run config. Do not accept an answer merely because it appears later in a wrong completion. Do not require answers to be a single token. Predeclare aliases before inspecting feature results; for the context controls, `yen` / `Japanese yen` is a reasonable explicit alias pair.

Report both **distinct facts** (at least one retained prompt) and **retained prompt rows**. Show raw -> clean -> model-correct -> discovery -> held-out counts and rejection reasons, by group and relation. A fact with five correct paraphrases must not receive five times the weight of a fact with one. Model agreement is not independent verification of a potentially bad source label.

If a group has fewer than two distinct discovery facts, report its examples but mark its feature set and overlap **insufficient data**, not zero overlap. Small groups remain exploratory; do not generate synthetic facts to equalize group sizes.

## S2 â Extract and independently select features

**Primary extraction position:** the last non-padding token of each retained prompt, immediately before generating its answer. This uses the model's normal state with all question context available. It is NOT necessarily the `Japan` token or a state before `is`; the early-position test is separate in S4. Save the chosen token index and text for every row.

For each layer, encode that position's native MLP input. A feature ID is `(layer_index, feature_index)`; identical feature indices in different layers are different features. Save sparse nonzero activations. For this ReLU dictionary, "active" means encoded activation `> 0`; keep actual values too.

For each group and feature:

1. Average its active/not-active indicator across the retained paraphrases of each fact, then average across distinct facts. Call this `positive_rate`.
2. Compute `background_rate` from correct relation/template-matched backgrounds, using the positive group's relation/template weights. Exclude unsupported cells from both rates and expose their counts; do not count missing cells as inactive.
3. Define the simple ranking score `positive_rate - background_rate`.

Require a positive score and activation in at least two distinct positive facts contributing to supported comparisons. Within each layer, select up to the top **20** qualifying features; break ties by `positive_rate`, then feature index. Keep all four groups' rates and mean activations in the output table for interpretation. Rank each group independently against its matched background; **do not use another target group as its negative class**.

**Do not define Tokyo features as "high on Tokyo, low on Japan" before measuring overlap.** That would bias the requested intersection toward zero. A feature may qualify for both sets. Do not choose thresholds by the resulting overlap or by held-out capital behavior.

Save the full qualifying ranking and also compute top-10 and top-50 sets from the same cached scores as a small cutoff check, with no new model runs. Report the main top-20 result first. Do not force 20 features if fewer qualify. For a compact inspection panel, show up to five highest-ranked candidates per entity, with layer/ID, rates across all four groups, and actual activating examples. Treat entity names as provisional labels.

## S3 â Report overlap

Let `F_Tokyo`, `F_Japan`, `F_Berlin`, and `F_Germany` be the selected sets of exact `(layer, feature)` IDs.

For each pair report:

- Number in each set, shared count, and unique-to-each count.
- Shared / first-set size and shared / second-set size, with denominators shown.
- Jaccard overlap = shared count / number in the union.

Use `N/A` for undefined ratios or insufficient-data groups. If adequate data produce an empty set, state that separately. Report per-layer and all-scanned-layer results; label the latter as a pooled dictionary-ID result, not a model-independent semantic overlap.

Headline comparisons: **Tokyo vs Japan** and **Berlin vs Germany**. Include all six pairwise comparisons in a small four-by-four heatmap so Tokyo/Berlin and Japan/Germany provide context. Save the shared/unique feature lists, not only percentages. Also state that shared feature IDs do not imply the same activation magnitude; show each shared candidate's group rates. Different IDs do not guarantee independent semantic directions.

Required figures (reuse cached data):

1. `plots/sample_counts.png`: retained facts and prompts per primary group, clearly distinguished.
2. `plots/feature_overlap.png`: four-set Jaccard heatmap plus the two headline pairs' exact counts.
3. `plots/candidate_activations.png`: a small feature-by-group rate table/heatmap for the inspected candidates.

Each reported metric must be defined in `REPORT.md` Methods and represented in the corresponding figure/table. No p-values, bootstrap intervals, or new statistical machinery in this first pass.

## S4 â Check transfer to the early capital position

Without reselecting features, evaluate:

- `The capital of Japan` and `The capital of Japan is`.
- `The capital of Germany` and `The capital of Germany is`.

Locate the country span using tokenizer offsets; use its last subtoken and record tokenization. First verify the baseline predicts the intended city after the fixed bridge ` is`, and record the prefix's top next token and the bridge likelihood. If `is` is not the top continuation, report that; do not describe the example as "same top-1 next token" without checking.

At the country position **before `is`**, record which members of each frozen feature set are naturally active, their activation values, and their shared/unique status. Causal attention permits a single full forward on the `... country is` input while reading the earlier country position; verify this agrees with the prefix-only run within numerical tolerance.

Save `early_target_features.csv` and a compact panel in `candidate_activations.png`. A feature found immediately before a Tokyo answer may be absent before `is`; that is a result, not a reason to inject it or tune the selection. The primary overlap result and early-position activation result answer different questions.

## S5 â Small conditional suppression check

Only after S1-S4 are saved, if budget permits and early-active city candidates exist, test the discussed intervention. Otherwise record why this stage was skipped.

- For each city, start with candidates in `F_city` but not `F_country` that are naturally active at its held-out country position. Call these **set-exclusive candidates**, not "country-free information". Choose one layer from `0..26` by highest total discovery score among eligible candidates, with layer-index tie breaking; do not choose it by intervention outcome. Use up to five candidates at that layer.
- In the original Qwen forward, modify only the country position's MLP output:

  `y_new = y - alpha * sum_i(a_i * d_i)`, for `alpha = 0, 0.5, 1`.

  Compute `a_i` from that run's natural MLP input; `d_i` is the corresponding decoder output direction. Preserve the original MLP output/reconstruction error. Continue all later layers and recompute downstream states; do not edit a cached tensor after the forward or reuse an unmodified KV cache. Layer 27 has no later attention to carry a country-position MLP edit to the `is` position, so exclude it here.
- Teacher-force the same bridge ` is`. Record the target city's next-token probability when it is one token, or full teacher-forced answer log-probability otherwise, plus the generated continuation. Separately record `is` likelihood at the country prefix. `alpha=0` must reproduce baseline.
- Compare with a same-layer naturally active feature set outside both entity sets, matched as closely as possible in feature count and perturbation L2 norm; report achieved norm matching. Restore the original output to check recovery. These are practical controls, not proof of perfect selectivity.
- Run the same frozen edit rule on the country-context controls, documenting their edited positions. Their preserved behavior is useful collateral evidence, **not proof that all country information survived in the same capital-prompt state**. Do not claim that stronger result without a validated same-state country readout.
- Save one dose-response figure/table. If no set-exclusive early candidates survive, report this; do not change the overlap definition or search indefinitely.

## Deliverables

- `REPORT.md`, `RESULTS.md`: concise, current results, limitations, and a plain-language verdict.
- `run_config.json`: exact source/model/transcoder revisions, environment, layers, tokenizer/decoding settings, seed, aliases, feature thresholds, and position definitions.
- `samples.jsonl`: group, role, fact ID, relation, subject, answer, source, template, prompt, correctness, completion, rejection reason, and token indices. Include backgrounds and rejected rows for audit.
- `sample_counts.csv`: counts by group/relation/stage, with facts and prompts separate.
- `feature_rankings.csv` and `feature_sets.json`: scores, support, selected IDs, and cutoffs.
- `feature_overlap.csv` and shared/unique ID lists; `early_target_features.csv`.
- `plots/` with the required figures; suppression outputs only if run.
- Minimal reproducible scripts and a single documented rerun command; source-data license notice; journal/changelog updates.

## Stages (checklist)

- [x] S0 â Read operator rules; validate pinned model, hooks, one transcoder, and resource plan.
- [x] S1 â Build candidate/background groups; run correctness filtering; save all sample counts.
- [x] S2 â Extract features; independently select four sets; save scores and examples.
- [x] S3 â Produce exact overlap counts, ratios, feature lists, and figures.
- [x] S4 â Check frozen candidates at the held-out pre-`is` country positions.
- [x] S5 â Complete the small suppression check or explicitly record why it was skipped.
- [x] Finalize current-best report/results and write `STOP`.

## Out of scope (do NOT)

- Do not train a new transcoder/SAE, change model checkpoints, build a full attribution graph, or launch large interpolation sweeps.
- Do not define exclusive city/country sets before computing overlap, conflate prompt counts with fact counts, or pool feature IDs across layers without layer labels.
- Do not label all Japan-containing prompts as one homogeneous class, treat `German` as the country, or combine country-context and country-answer samples without explicit stratification.
- Do not infer planning, pure semantic localization, or preservation of all country information from activation/overlap alone.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with: `On track? <yes/no> â <stage, % done, blocker if any>`.

## Current status

Complete. S0–S5 were run on all 28 layers; `REPORT.md` and `RESULTS.md` hold the results and all machine-readable outputs are in `results/`. Discovery facts / prompts: Tokyo 6 / 18, Japan 16 / 21, Berlin 2 / 7, Germany 10 / 10. Top-20 sets: 276 / 403 / 85 / 271 features; Tokyo–Japan share 58 (Jaccard 0.093), Berlin–Germany share 14 (0.041). Strong features (score >= 0.5) are mostly shared within a country (16 / 20 Tokyo features are Japan features). A few shared features are active on the country token before `is`. The suppression check was run and is null.

## Next step

None. Success criterion met; `STOP` written.

## Source references

- Model: https://huggingface.co/Qwen/Qwen3-0.6B
- Transcoders and configuration: https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0
- Loader: https://github.com/decoderesearch/circuit-tracer/blob/0968d11a0e14a51d2fa4c445466949d2a93a5896/circuit_tracer/transcoder/single_layer_transcoder.py
- LRE data: https://github.com/evandez/relations/tree/1b9ec3cf2b8368e42bde7e80fcef384312a6ec07/data/factual
- Transcoder method: https://arxiv.org/abs/2406.11944
