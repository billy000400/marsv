# JOURNAL — Direction: TODO — describe this direction

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-09-23 — iter 1: S1 sanity check done, S2 sweep launched

**Assumptions (loop mode, logged per CLAUDE.md §1):**
- "Layer 0" = the embedding layer (HF `hidden_states[0]` = wte + wpe). The position embedding at
  the last position is shared by A and B, so interpolating wte[A]→wte[B] via `inputs_embeds` is
  identical to interpolating layer-0 hidden states, and D = ||wte[big] − wte[B]||.
  Verified: a direct `input_ids` forward and the `inputs_embeds` endpoint agree exactly (max diff 0.0).
- "Block 18" = output of `transformer.h[18]` (0-indexed; = `hidden_states[19]`), matching dir21's
  convention. "Final block" = output of `transformer.h[35]` **before** `ln_f` (hook). Rejected:
  `hidden_states[-1]`, which has ln_f applied and is therefore not a block output.
- Readout = last position only (the only position the edit affects).
- W uses the FIRST linear-interpolated crossing of 0.1 and of 0.9. Flag if either level is crossed
  more than once or if y(t) ever steps back by > 0.05 (overshoot above 1, then return).
- All 50,256 other tokens included (incl. `<|endoftext|>`); nothing needed excluding.

**S1 result** (`results/s1_sanity.json`, `plots/s1_sanity_curves.png`, 20 tokens): y(0)=0, y(1)=1
exactly; all curves are sigmoid-like and the marked t_0.1/t_0.9 points sit on the curves. Close
adjectives (` huge`, ` large`, ` small`, ` tiny`, ` bigger`: D 1.2–1.8) give wide transitions
(W_final 0.33–0.56); unrelated words/punctuation (D 1.6–2.4) give sharp ones (W_final 0.05–0.12).
Note ` the` (D=1.61) is sharp even though D is small, so D alone may not decide it. Block 35 is
sharper than block 18 in every case. A few final-layer curves overshoot y>1 (` Paris`, `!!`) → flagged.

**S2**: full sweep running (~30 min at 29 tok/s, fp32), resumable shards in results/sweep/.

Next: aggregate the shards → CSV; make Fig 1a/1b, 2a/2b, and representative curves; write REPORT.md.

On track? yes — S1 done, S2 running (~25% of plan), no blocker.
