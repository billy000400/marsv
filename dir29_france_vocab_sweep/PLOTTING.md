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

The current style copies panel (a) of
`../dir27_interp_dist_vs_plateau_width/plots/block0_resid_post/fig2_final_width_distribution.png`.
That figure belongs to Direction 27, the sibling project folder `../dir27_interp_dist_vs_plateau_width/`, which
measured the same transition width for other token pairs. The copied style is:
one 6 x 4.5 in panel, 200 bins, linear y-axis, default matplotlib styling. The title is
`"The capital of [France]" -> [B] sweep`.

## Data

`results/france_tokens.csv` is written by `experiments/analyze.py` from the raw sweep output. The sweep
(`experiments/sweep.py`) saves its results in 25 pieces called shards, each covering up to 2,048 consecutive token IDs,
as `results/sweep/shard_*.npz`. Re-run
`analyze.py` only if the sweep itself changes; a style change never needs it. `W_final` is the transition
width at the final block (block 35) of GPT-2 Large, as defined in `REPORT.md` Methods.

If the caption needs to change with the style (for example a log axis), update the Figure 1 caption in
`REPORT.md` and `RESULTS.md` too.

## Colour rules

The main reader of these plots has red-green colour deficiency, so every restyle must stay readable without
telling red from green (the full rule is rule 13 of the project-wide instructions file `../CLAUDE.md`):

- Never use red against green as a contrast.
- For separate series use these colours in this order: `#0072B2` (blue), `#D55E00` (vermillion), `#CC79A7`
  (reddish purple), `#56B4E9` (sky blue), `#E69F00` (orange). Use at most five.
- If the plot has more than one series, also give each a different line style, marker or hatch, so the figure
  still reads in grayscale.
- For colour maps use `viridis` or `cividis` (one direction) or `coolwarm` / `RdBu` (two directions). Do not use
  `jet`, `rainbow`, `hsv`, `RdYlGn` or `Spectral`.
- In captions, name a series by what it is and by its line style or marker, never by its colour.
