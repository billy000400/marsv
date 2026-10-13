# Full-vocabulary scan timing per model

How long one full-vocabulary sweep took for each model in this direction. A sweep interpolates from
` big` to every other token in the tokenizer and records the plateau width at the middle and final
block. Times are the sum of the per-shard `total` timings in the sweep logs, cross-checked against
the shard file timestamps. They cover the scan only: model download, plotting and report writing are
excluded.

| Model | Precision | Tokens scanned | Scan time | Throughput | Run date | Log |
|---|---|---|---|---|---|---|
| GPT-2 Large | fp32 | 50,256 | 25.7 min | 33 tok/s | 2026-09-23 | `results/s2_sweep.log` |
| GPT-2 XL | fp32 | 50,256 | 16.5 min | 51 tok/s | 2026-09-28 | `results/gen/sweep_gpt2-xl.log` |
| Pythia-1.4B | fp32 | 50,276 | ~27.5 min | 31 tok/s | 2026-09-28 | `results/gen/sweep_pythia-1.4b.log` |
| Qwen3-8B base | bf16 | 151,668 | 41.5 min | 61 tok/s | 2026-09-28 | `results/gen/sweep_qwen3-8b-base.log` |
| Llama-3.1-8B | bf16 | 128,255 | 32.1 min | 67 tok/s | 2026-09-28 | `results/gen/sweep_llama-3.1-8b.log` |

Notes:

- The Pythia-1.4B time is partly estimated. Its log lacks the first shard (4,096 tokens), so 25.3 min
  of logged time was extended by about 2.2 min at the rate of the other shards. The shard timestamps
  also show about 2 min of extra wall time, which looks like a restart.
- The four generalization models ran back to back on 2026-09-28 from 00:31 to 02:40, about 2 h 9 min
  of wall time. That includes about 9.5 min between GPT-2 XL finishing and Qwen3 starting, most likely
  spent staging the 8B weights.
- The GPT-2 Large sweep used 2,048-token shards; the other four used 4,096-token shards.
