# RESULTS — Does the GPT-2 Large distance-vs-width result generalize? (GPT2-XL, Pythia 1.4B, Llama-3.1-8B, Qwen3-8B-Base)

This file holds the detailed evidence behind `REPORT_generalization.md`. The GPT-2 Large study it
extends is in `REPORT.md` and `RESULTS.md`; those files are unchanged. All definitions (interpolation
h(t), input distance D, progress y(t), transition width W = t_0.9 − t_0.1, the non-monotonic flag) are
exactly those of `REPORT.md` and are restated in the Methods of `REPORT_generalization.md`.

## G0 — Setup and checks

**Models and runs.** Prompt "The house was big", A = ` big` (the last token in every tokenizer), B =
every other token ID the tokenizer defines (rows of the embedding matrix beyond the tokenizer, which
are padding, are skipped). 101 steps of t. Readouts: last-position output of the middle block
(n_blocks/2, counted from 0) and the final block (before the final norm).

| Model | HF repository | blocks (mid, final) | tokens B | weights / arithmetic |
|---|---|---|---|---|
| GPT2-XL | `openai-community/gpt2-xl` | 48 (24, 47) | 50,256 | float32 |
| Pythia 1.4B | `EleutherAI/pythia-1.4b` | 24 (12, 23) | 50,276 | float32 |
| Llama-3.1-8B | `unsloth/Meta-Llama-3.1-8B` (ungated copy of `meta-llama/Llama-3.1-8B`) | 32 (16, 31) | 128,255 | bfloat16 weights and matrix products, float32 residual stream |
| Qwen3-8B-Base | `Qwen/Qwen3-8B-Base` | 36 (18, 35) | 151,668 | bfloat16 weights and matrix products, float32 residual stream |

**Computation.** The first tokens of the prompt do not depend on B, so their attention keys and values
are computed once and reused; only the last position is run for each of the 101 × (number of B)
inputs (`experiments/gen_common.py`, `experiments/gen_sweep.py`). For the 8B models, blocks that do not
fit the 14.4 GB GPU budget (Llama blocks 22–31, Qwen3 blocks 24–35) are held in CPU memory and copied to
the GPU for each forward pass. This is exact; no weights are quantized. Figures and tables:
`experiments/gen_plots.py`.

**Checks.** Each sweep first compares the reused-prefix run against a full run of the whole prompt.
For the float32 models the block outputs agree to a relative difference below 3e-5. For the bfloat16
models the two runs round differently. The per-model sections give the resulting median and largest
|ΔW| over 32 random tokens; it is small next to the width differences discussed. An earlier Qwen3
attempt that also kept the residual stream in bfloat16 gave about twice the rounding difference, so the
residual stream is kept in float32.

## G1 — GPT2-XL

**Run.** 50,256 tokens B; middle block 24, final block 47 of 48;
prompt token IDs [464, 2156, 373, 1263]. Reused-prefix run vs full-prompt run (3 tokens): largest relative difference of block outputs 7.3e-06. Per-token values: `results/gen/gpt2-xl.csv`; all numbers below:
`results/gen/gpt2-xl_summary.json`.

**Headline numbers.** Median D 2.344 (5th–95th percentile 1.998–2.680).
Median W_mid 0.174, median W_final 0.089; W_final < W_mid for
95.3% of tokens. Flagged: 11.6% of tokens
(11.5% at the final block). Tallest W_final histogram bin (200 bins):
0.063–0.067 (1,532 tokens). W_final 90th / 99th percentile:
0.186 / 0.373. W_final of B = ` large`: 0.502.

**Separate group.** 57 byte-level tokens (shown as `�`) sit at D ≈ 3.97 with nearly identical widths
(median W_mid 0.373, W_final 0.182). This mirrors the 64 rarely seen tokens at D ≈ 5.17 in GPT-2 Large.

The table below summarizes the width-vs-distance cloud by D quintile (five equal-size token groups). It
is a description of the cloud, not a statistical test:

| D range (quintile) | tokens | median W_mid | median W_final |
|---|---|---|---|
| 1.172–2.166 | 10,052 | 0.167 | 0.086 |
| 2.166–2.289 | 10,052 | 0.173 | 0.090 |
| 2.289–2.399 | 10,052 | 0.174 | 0.084 |
| 2.399–2.524 | 10,052 | 0.175 | 0.087 |
| 2.524–3.991 | 10,052 | 0.182 | 0.101 |

The S4-style table lists, for each W_final band, the token count and the seven tokens with the lowest D
(examples only; `␣×n` = n spaces):

| W_final band | tokens | lowest-D examples |
|---|---|---|
| < 0.05 | 9,601 | ` a`, ` an`, ` to`, ` it`, ` that`, ` for`, ` one` |
| 0.05–0.10 | 19,303 | ` the`, ` Big`, ` in`, `,`, ` more`, ` on`, ` of` |
| 0.10–0.15 | 12,332 | ` great`, `big`, ` good`, ` many`, ` high`, ` would`, ` about` |
| 0.15–0.20 | 4,984 | `,"`, ` long`, ` important`, ` strong`, ` makes`, ` lots`, ` old` |
| 0.20–0.30 | 2,899 | ` major`, ` new`, ` little`, ` BIG`, ` nice`, ` extremely`, ` crazy` |
| 0.30–0.50 | 1,051 | ` bigger`, ` massive`, ` small`, ` gigantic`, ` enormous`, ` larger`, ` significant` |
| ≥ 0.50 | 86 | ` huge`, ` large`, ` biggest`, ` giant`, ` hefty`, ` largest`, ` tremendous` |

Widest 15 tokens (W_final): `UGE` (0.73), ` giant` (0.69), ` fatty` (0.67), ` monstrous` (0.65), `quished` (0.65), ` tremendous` (0.64), `arger` (0.64), ` externalToEVAOnly` (0.64), `huge` (0.63), ` myriad` (0.61), ` makeshift` (0.61), ` vast` (0.60), `tiny` (0.60), ` immense` (0.60), `imsy` (0.60).

Figures: `plots/gen/gpt2-xl_fig1_distance_vs_width.png`, `plots/gen/gpt2-xl_fig2_zoom_distance_vs_width.png`,
`plots/gen/gpt2-xl_fig3_final_width_distribution.png` (shown in `REPORT_generalization.md`, Figures 1–3).

## G2 — Pythia 1.4B

**Run.** 50,276 tokens B; middle block 12, final block 23 of 24;
prompt token IDs [510, 2419, 369, 1943]. Reused-prefix run vs full-prompt run (3 tokens): largest relative difference of block outputs 2.7e-05. Per-token values: `results/gen/pythia-1.4b.csv`; all numbers below:
`results/gen/pythia-1.4b_summary.json`.

**Headline numbers.** Median D 1.390 (5th–95th percentile 1.350–1.421).
Median W_mid 0.423, median W_final 0.328; W_final < W_mid for
95.8% of tokens. Flagged: 0.2% of tokens
(0.2% at the final block). Tallest W_final histogram bin (200 bins):
0.327–0.331 (1,153 tokens). W_final 90th / 99th percentile:
0.417 / 0.507. W_final of B = ` large`: 0.730.

**Separate group.** 237 tokens sit at D < 1.05 (D ≈ 1.0), far from the main cloud at D ≈ 1.39, with median
W_final 0.191. They are runs of spaces, tabs and newlines, byte tokens and `<|padding|>`.

The table below summarizes the width-vs-distance cloud by D quintile (five equal-size token groups). It
is a description of the cloud, not a statistical test:

| D range (quintile) | tokens | median W_mid | median W_final |
|---|---|---|---|
| 0.935–1.372 | 10,056 | 0.435 | 0.341 |
| 1.372–1.385 | 10,056 | 0.426 | 0.332 |
| 1.385–1.396 | 10,057 | 0.423 | 0.328 |
| 1.396–1.406 | 10,056 | 0.419 | 0.324 |
| 1.406–1.471 | 10,056 | 0.413 | 0.317 |

The S4-style table lists, for each W_final band, the token count and the seven tokens with the lowest D
(examples only; `␣×n` = n spaces):

| W_final band | tokens | lowest-D examples |
|---|---|---|
| < 0.05 | 0 |  |
| 0.05–0.10 | 10 | `\r\n\t\t\t\t\t`, `\r\n\t\t\t`, `\n\n\t\t\t`, `\n\n\t`, `\r\n\r\n\t`, ` \n\t`, `\r\n\t` |
| 0.10–0.15 | 15 | `␣×9`, `␣×11`, `\r\n␣×8`, `\n\t`, `\r\n\t\t\t\t`, `\n\t\t\t`, `  \n` |
| 0.15–0.20 | 360 | `\n␣×41`, `␣×11\n `, `\n\n␣×17`, `␣×76`, `\n␣×15`, `\n␣×4\n   `, `\n␣×8` |
| 0.20–0.30 | 16,127 | `\n␣×50`, `␣×73`, `␣×12`, `\n␣×26`, `\n␣×34`, `␣×16`, `␣×32` |
| 0.30–0.50 | 33,147 | ` Big`, ` BIG`, `14514500`, ` careg`, `␣×8`, `big`, `Big` |
| ≥ 0.50 | 617 | ` bigger`, ` huge`, ` biggest`, ` large`, ` gigantic`, ` of`, `.` |

Widest 15 tokens (W_final): ` huge` (0.75), ` giant` (0.74), ` large` (0.73), ` gigantic` (0.72), ` enormous` (0.72), ` immense` (0.72), ` massive` (0.72), ` vast` (0.71), ` monstrous` (0.69), ` tremendous` (0.68), ` larger` (0.67), ` largest` (0.67), `sized` (0.67), ` mega` (0.65), ` bulky` (0.64).

Figures: `plots/gen/pythia-1.4b_fig1_distance_vs_width.png`, `plots/gen/pythia-1.4b_fig2_zoom_distance_vs_width.png`,
`plots/gen/pythia-1.4b_fig3_final_width_distribution.png` (shown in `REPORT_generalization.md`, Figures 4–6).

## G3 — Llama-3.1-8B

**Run.** 128,255 tokens B; middle block 16, final block 31 of 32;
prompt token IDs [128000, 791, 3838, 574, 2466]. Precision check (32 random tokens, reused-prefix run vs full-prompt run): median |ΔW| 0.0009 (middle) and 0.0019 (final); largest 0.002 and 0.008. Per-token values: `results/gen/llama-3.1-8b.csv`; all numbers below:
`results/gen/llama-3.1-8b_summary.json`.

**Headline numbers.** Median D 0.909 (5th–95th percentile 0.809–0.973).
Median W_mid 0.279, median W_final 0.203; W_final < W_mid for
88.8% of tokens. Flagged: 7.7% of tokens
(7.5% at the final block). Tallest W_final histogram bin (200 bins):
0.192–0.195 (2,467 tokens). W_final 90th / 99th percentile:
0.308 / 0.423. W_final of B = ` large`: 0.668.

**Separate group.** 475 tokens sit at D ≈ 0.606 (0.600–0.612): the 256 special and reserved tokens
(IDs 128000–128255) and byte-level tokens (shown as `�`). Median W_mid 0.402, W_final 0.389; 81% flagged.
They make up 298 of the 682 tokens in the histogram spike at W_final ≈ 0.39.

**Widest ASCII word tokens.** ` large` (0.67, widest of all tokens), ` chubby`, ` snork`, ` defiance`,
` oversized`, ` huge` (0.61), ` demeanor`, ` alertController`, ` neoliberal`, ` staunch`, ` malicious`,
` substantial` (0.57), ` massive` (0.57). Core (D > 0.8): W_final < W_mid for 89.1%.

The table below summarizes the width-vs-distance cloud by D quintile (five equal-size token groups). It
is a description of the cloud, not a statistical test:

| D range (quintile) | tokens | median W_mid | median W_final |
|---|---|---|---|
| 0.582–0.867 | 25,651 | 0.280 | 0.199 |
| 0.867–0.898 | 25,651 | 0.276 | 0.198 |
| 0.898–0.918 | 25,651 | 0.278 | 0.204 |
| 0.918–0.940 | 25,651 | 0.278 | 0.208 |
| 0.940–1.125 | 25,651 | 0.281 | 0.205 |

The S4-style table lists, for each W_final band, the token count and the seven tokens with the lowest D
(examples only; `␣×n` = n spaces):

| W_final band | tokens | lowest-D examples |
|---|---|---|
| < 0.05 | 123 | `\u200cهاي`, `만남`, `.\n\n`, ` กรกฎาคม`, `<\|begin_of_text\|>`, ` ขนาด`, ` افزار` |
| 0.05–0.10 | 5,408 | ` Big`, `џџџџџџџџџџџџџџџџ`, `џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ`, ` ослож`, ` 首页第`, `џџџ`, `醴醴` |
| 0.10–0.15 | 20,386 | `ılmış`, `▍▍▍▍▍▍▍▍`, ` -->\r\n\r\n`, `webElementProperties`, `тися`, ` зустрі`, ` vyrá` |
| 0.15–0.20 | 36,004 | `lıkları`, `abancı`, `üyordu`, `jícím`, ` 神马收录`, `artisanlib`, ` vzdál` |
| 0.20–0.30 | 51,587 | `itelné`, `kyně`, ` přibliž`, `();\r\r\n`, `quotelev`, ` předsed`, `iyesi` |
| 0.30–0.50 | 14,583 | `ımda`, ` характериз`, `ючись`, `дивиду`, ` располаг`, `ılmaz`, `sahuje` |
| ≥ 0.50 | 164 | ` large`, ` เพราะ`, ` huge`, ` size`, `вают`, `ukarı`, `μως` |

Widest 15 tokens (W_final): ` large` (0.67), ` chubby` (0.65), ` snork` (0.63), ` defiance` (0.62), ` oversized` (0.61), ` huge` (0.61), ` demeanor` (0.61), ` alertController` (0.61), ` neoliberal` (0.60), `่อย` (0.59), ` staunch` (0.59), `esseract` (0.59), `�게` (0.59), ` malicious` (0.59), ` egy` (0.58).

Figures: `plots/gen/llama-3.1-8b_fig1_distance_vs_width.png`, `plots/gen/llama-3.1-8b_fig2_zoom_distance_vs_width.png`,
`plots/gen/llama-3.1-8b_fig3_final_width_distribution.png` (shown in `REPORT_generalization.md`, Figures 7–9).

## G4 — Qwen3-8B-Base

**Run.** 151,668 tokens B; middle block 18, final block 35 of 36;
prompt token IDs [785, 3753, 572, 2409]. Precision check (32 random tokens, reused-prefix run vs full-prompt run): median |ΔW| 0.0009 (middle) and 0.0032 (final); largest 0.003 and 0.105. Per-token values: `results/gen/qwen3-8b-base.csv`; all numbers below:
`results/gen/qwen3-8b-base_summary.json`.

**Headline numbers.** Median D 2.329 (5th–95th percentile 1.930–2.495).
Median W_mid 0.233, median W_final 0.230; W_final < W_mid for
56.2% of tokens. Flagged: 21.3% of tokens
(21.2% at the final block). Tallest W_final histogram bin (200 bins):
0.190–0.194 (2,665 tokens). W_final 90th / 99th percentile:
0.374 / 0.534. W_final of B = ` large`: 0.627.

**Low-D edge.** Tokens with D < 1.93 (7,558, 5.0%) are almost all non-English pieces (Korean syllables,
Thai, Hebrew and Arabic fragments, emoji, CJK). D < 1.9: 4,245 tokens, median W_final 0.136, W rises with
D along a curved ramp (`REPORT_generalization.md` Figure 10). 1.9 ≤ D < 1.93: 3,313 tokens, median W_final 0.266, 48% flagged. The
core D ≥ 2.1 (133,906 tokens): median W_mid 0.233, W_final 0.233, 21.6% flagged, W_final < W_mid for
54.3%.

**Size words.** ` large`: D 2.348, W_mid 0.648, W_final 0.627 (rank 281 of 151,668). ` huge`: W_final
0.558. ` giant`: 0.615 (flagged). ` small`: 0.412. ` the`: 0.215. Widest ASCII word tokens: ` sorted`
(0.71), ` addslashes` (0.71), `wright`, ` quaint`, ` textDecoration`, `Frozen`, ` tremendous` (0.68),
` ForCanBeConverted`, `seller`, ` widest` (0.64).

The table below summarizes the width-vs-distance cloud by D quintile (five equal-size token groups). It
is a description of the cloud, not a statistical test:

| D range (quintile) | tokens | median W_mid | median W_final |
|---|---|---|---|
| 1.822–2.199 | 30,334 | 0.239 | 0.215 |
| 2.199–2.297 | 30,333 | 0.243 | 0.244 |
| 2.297–2.360 | 30,334 | 0.235 | 0.238 |
| 2.360–2.424 | 30,333 | 0.228 | 0.232 |
| 2.424–2.664 | 30,334 | 0.217 | 0.220 |

The S4-style table lists, for each W_final band, the token count and the seven tokens with the lowest D
(examples only; `␣×n` = n spaces):

| W_final band | tokens | lowest-D examples |
|---|---|---|
| < 0.05 | 97 | `ﰞ`, `깞`, `�`, `𝄕`, `괙`, `쒯`, `𝆳` |
| 0.05–0.10 | 3,633 | `ꀰ`, `ניוזל`, `젡`, `퓷`, `넑`, `ספטמ`, `𦒍` |
| 0.10–0.15 | 17,683 | `띕`, `딮`, `뢉`, `옂`, ` zwłas`, `꾈`, ` thuisontvangst` |
| 0.15–0.20 | 33,505 | `�`, `툿`, ` paździ`, `췬`, `ꦨ`, `𨭉`, `듷` |
| 0.20–0.30 | 59,695 | `첧`, `뜷`, `뙝`, `둁`, `ﲒ`, `쒔`, `PostalCodesNL` |
| 0.30–0.50 | 34,660 | `땧`, ` الديمقرا`, ` 오�`, `굠`, `칕`, `ᨹ`, `쩻` |
| ≥ 0.50 | 2,395 | `ꌼ`, `퀅`, `뺃`, `𖥨`, `톢`, `格會員`, `촋` |

Widest 15 tokens (W_final): `ฝึก` (0.74), ` ดัง` (0.74), `น่าจะ` (0.74), `เห็นว่า` (0.73), `琢` (0.73), ` ซึ่งเป็น` (0.73), `ห์` (0.72), `亿元以上` (0.72), `突` (0.72), `ฉัน` (0.71), `ฝรั่ง` (0.71), `มหาวิทยาลัย` (0.71), `กิน` (0.71), `พฤติ` (0.71), ` sorted` (0.71).

Figures: `plots/gen/qwen3-8b-base_fig1_distance_vs_width.png`, `plots/gen/qwen3-8b-base_fig2_zoom_distance_vs_width.png`,
`plots/gen/qwen3-8b-base_fig3_final_width_distribution.png` (shown in `REPORT_generalization.md`, Figures 10–12).


## G5 — Comparison across models

Each row condenses one model's section; GPT-2 Large values are from `RESULTS.md` (S3, S4) and
`REPORT.md`. "Median W_final across D quintiles" is the range of the five quintile medians:

| Model | median W_final across D quintiles | median W_mid → W_final | final block sharper | final-block curves flagged | W_final peak | tokens in 0.30–0.50 | rank of ` large` by W_final |
|---|---|---|---|---|---|---|---|
| GPT-2 Large | flat core (0.109, 0.114 for D 2.0–3.0) | 0.195 → 0.112 | 97.7% | 5.3% | 0.09–0.10 | 787 | 53 |
| GPT2-XL | 0.084–0.101 | 0.174 → 0.089 | 95.3% | 11.5% | 0.06–0.07 | 1,051 | 84 |
| Pythia 1.4B | 0.317–0.341 | 0.423 → 0.328 | 95.8% | 0.2% | 0.33 | 33,147 | 3 |
| Llama-3.1-8B | 0.198–0.208 | 0.279 → 0.203 | 88.8% | 7.5% | 0.19–0.20 | 14,583 | 1 |
| Qwen3-8B-Base | 0.215–0.244 | 0.233 → 0.230 | 56.2% | 21.2% | 0.19 | 34,660 | 281 |

**Where each finding holds.** Distance does not visibly predict W in the bulk of any model. The final
block is sharper in GPT2-XL, Pythia and Llama but not in Qwen3-8B-Base. Every W_final distribution is
one broad peak with a tail, and its position is model-specific. Tight groups of rarely seen tokens with
near-identical widths: GPT-2 Large (64 tokens, D ≈ 5.17), GPT2-XL (57, D ≈ 3.97), Pythia (237, D ≈ 1.0),
Llama (475, D ≈ 0.606). Qwen3 has a low-D multilingual edge instead, which is not tight in W.

**Files.** Per-token CSVs `results/gen/<model>.csv` (token_id, token, D, W_mid, W_final,
flag_nonmonotonic); summaries `results/gen/<model>_summary.json`; raw shards with all y(t) curves
`results/gen/<model>/shard_*.npz`; run logs `results/gen/sweep_<model>.log`.
