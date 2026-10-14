# How to redraw the sweep histogram

This note is for people and agents who need to change the look of the sweep plot
(Figure 1 of `REPORT.md` and `RESULTS.md`).

| What | Where |
|------|-------|
| Plot script | `experiments/plot_sweep.py` |
| Data it reads | `results/france_tokens.csv` (one row per token B; the script uses the column `W_final`) |
| Figure it writes | `plots/france_vocab_width_histogram.png` |

## Command

Run from this direction's root (`dir29_france_vocab_sweep/`):

```
MPLBACKEND=Agg python3 experiments/plot_sweep.py
```

It takes a few seconds, needs no GPU and loads no model. It overwrites the PNG in place.

## Changing the style

Edit the `STYLE` block at the top of `experiments/plot_sweep.py` (figure size, DPI, number of bins, colour,
title, axis labels) and re-run the command. For changes beyond those settings, edit `plot()`.
`load_data()` only reads the CSV and does not need to change for a style edit.

The current style copies panel (a) of Direction 27's
`../dir27_interp_dist_vs_plateau_width/plots/block0_resid_post/fig2_final_width_distribution.png`:
one 6 x 4.5 in panel, 200 bins, linear y-axis, default matplotlib styling. The title is
`"The capital of [France]" -> [B] sweep`.

## Data

`results/france_tokens.csv` is written by `experiments/analyze.py` from the raw sweep shards. Re-run
`analyze.py` only if the sweep itself changes; a style change never needs it. `W_final` is the transition
width at the final block (block 35) of GPT-2 Large, as defined in `REPORT.md` Methods.

If the caption needs to change with the style (for example a log axis), update the Figure 1 caption in
`REPORT.md` and `RESULTS.md` too. Keep to the colour rules in `../CLAUDE.md` rule 13.
