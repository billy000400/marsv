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

## 2026-09-23 — iter 2: waiting on S2, plotting scripts ready

- `kill -0` / `sleep` loops in backgrounded Bash exit immediately in this sandbox, so a watcher
  wrongly reported the sweep as dead. I relaunched it, saw the original PID was still alive, and
  killed the duplicate (PID 55952) about 5 s later, before it wrote any shard. Shards are
  deterministic, so no data was affected. Use Monitor on the log instead.
- Partial-data observation (8,192 tokens): the D≈5.16 cluster is ~45 byte-level tokens (control
  chars, invalid-UTF-8 fragments) that barely occur in training; all have W_final≈0.33.
- New `experiments/s5_curves.py` (Figure 3). s3_plots.py now draws unflagged points on top of
  flagged ones so they are visible.

## 2026-09-23 — iter 2 (cont.): S2–S5 done, direction complete

- Sweep finished: 50,256 tokens, 5.5% flagged (2,686 at the final layer, 163 at block 18).
- D vs W (Fig 1): the bulk (D 2–3, 97% of tokens) shows no visible slope at either layer. ` huge` is
  the nearest token (D=1.22) yet one of the widest (W_final 0.56). The widest tail = size words plus
  known under-trained tokens (` guiName`, `Downloadha`).
- Correction: the D>4 cluster (64 tokens) is not only byte tokens. It also includes known
  under-trained scraped strings (` RandomRedditor`, `StreamerBot`, ` externalToEVA`). The reports
  say "rarely seen tokens".
- W_final: one peak around 0.09–0.10 with a long tail; the sorted curve has no shelves. The only spike
  is at 0.33 (the D>4 cluster; 43 of 65 in the tallest bin).
- Median-by-D table kept as a descriptive summary, not a test (PLAN forbids correlation stats).
- check_render passes. Success criterion met → STOP.

## 2026-09-23 — feedback triage: human_feedback.txt
- Request: zoomed version of Figure 1 with x in [1.6, 3.3] and the matching y range.
- Routing: "Figure 1" = REPORT.md Figure 1 (D vs W scatter); x range is in D units, so RESULTS.md's
  Figure 1 (sanity curves, x = t) is ruled out. New output plots/fig1_zoom_distance_vs_width.png, added
  after Figure 1 in REPORT.md and RESULTS.md; original Figure 1 kept. "Corresponding y range" = min–max
  of W among tokens with D in [1.6, 3.3], per panel. Manifest set to ready.

## 2026-09-23 — feedback work: zoomed Figure 1
- Added `zoom()` to s3_plots.py (`python s3_plots.py zoom`), reading sweep.csv to avoid regenerating the
  frozen outputs; verified sha256 of sweep.csv and fig1–fig3 unchanged.
- Zoom holds 50,164/50,256 tokens (excluded: 9 with D<1.6, 19 with 3.3<D<4, 64 at D≈5.17). The core shows no
  clear slope at either layer, consistent with Finding 1. Embedded in REPORT.md (Fig 2) and RESULTS.md (Fig 3);
  check_render passes. Manifest -> review_pending.
