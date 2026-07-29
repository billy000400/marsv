"""Insert a visible, numbered caption line under every embedded figure (CLAUDE.md rule 12).

Alt text does not render, so each `![...](plots/x.png)` needs a `**Figure N.** ...` paragraph
immediately below it. Figures are numbered in the order they appear in the file. Idempotent: a
figure that already has a caption line below it is renumbered, not duplicated.

usage: python add_captions.py FILE [FILE ...]
"""
import re
import sys

CAPTIONS = {
    "candidate_prevalence_by_layer.png":
        "`A|C|B` rate under the frozen rule. **Left:** percentage of eligible paths with a "
        "persistent third top-1 token, per interpolation block of GPT-2 Large (x: block L; y: % of "
        "eligible paths). **Right:** the same rate split by which context supplied the tokens the "
        "patched activation runs inside (“cond. A” / “cond. B”) and for each "
        "control condition. Error bars are 95% Wilson intervals. The self-pair bar is exactly zero "
        "because a constant path has no eligible endpoints.",
    "top_candidate_probability_paths.png":
        "Next-token probability of the three named tokens along the path. x: interpolation "
        "coefficient α (0 = context A's activation, 1 = context B's); left y: probability of "
        "A (solid, circles), C (dashed, squares) and B (dash-dot, triangles); right y: "
        "Jensen–Shannon divergence between neighbouring α in bits (dotted grey). The "
        "shaded band is the detected C run. **Top row:** the three highest-scoring candidates. "
        "**Bottom row:** three candidates drawn at random from the qualifying set — the "
        "contrast between the rows is the point.",
    "segment_width_margin_distribution.png":
        "What a typical candidate looks like, over all 1,290. **Left:** C-segment width as a "
        "fraction of the 50-point α grid. **Middle:** minimum dominance margin of C over both "
        "endpoint tokens (probability units). **Right:** entry α versus exit α of the C "
        "run, with the dashed diagonal marking zero width. Most segments are 3–5 grid points "
        "wide and win by less than 0.05.",
    "matthew_dt_frozen.png":
        "The same six paths in output geometry. x: interpolation coefficient t (0 = A's "
        "activation, 1 = B's); y: relative output distance d(t) on the final logits, 0 = the output "
        "looks like endpoint A, 1 = like endpoint B. The dashed grey line is the no-plateau "
        "reference d = t; the shaded band is the detected C run; thin vertical lines mark every "
        "top-1 token change. **Top row:** the three highest-scoring candidates (staircases). "
        "**Bottom row:** three random candidates (not staircases). Titles give the block, the C "
        "run's α range and its flatness ρ.",
    "subplateau_dwell.png":
        "Is the third region a shelf? **Left:** distribution of the C-window flatness ρ "
        "(x: ρ, range of d ÷ width in t, clipped at 6; y: density) for the 1,290 "
        "candidates (solid) and for the same α windows on 1,290 matched non-candidate paths "
        "(dashed); dashed rule at ρ = 1, dotted rule at the post-hoc ρ = 0.5 cut. "
        "**Middle:** median ρ (circles) with inter-quartile range (hatched) against the decile "
        "of the frozen candidate score (10 = highest). **Right:** histogram of the mean output "
        "distance across the C run. Read it as: the median candidate is *not* a plateau, but the "
        "pre-frozen score still ranks the flat ones highest.",
    "matthew_dt_gallery.png":
        "The flat tail: the six candidates with the flattest C windows (post-hoc selection, "
        "ρ < 0.5 and C run ≥ 5 grid points, out of 1,290). Axes as in Figure 4 — "
        "x: interpolation coefficient t; y: relative output distance d(t); dashed grey = the "
        "no-plateau diagonal; shaded band = the C run; thin vertical lines = top-1 token changes. "
        "Every panel is flat, jump, flat, jump, flat — the sub-plateau shape the MNIST work "
        "predicted.",
    "c_region_confidence.png":
        "Is the third region confident? x: top-1 probability (left panel) and predictive entropy in "
        "bits (right panel); y: number of candidate paths. Solid = measured at the centre of the C "
        "region, dashed = the mean of the two path endpoints, over all 1,290 candidates. The C "
        "region is *less* peaked and *higher* entropy than the endpoints, which argues against "
        "reading it as an extra confident state.",
    "intermediate_token_census.png":
        "Which tokens play the third role. **Left:** the 15 commonest intermediate (C) tokens over "
        "the 1,290 candidate paths. **Right:** the 15 commonest endpoint (A) tokens over the 7,611 "
        "eligible paths. x: number of paths; y-axis labels are the decoded token strings. The two "
        "lists are drawn from the same generic high-frequency pool.",
    "threshold_sensitivity.png":
        "Robustness of the headline rate to the two frozen thresholds. x: persistence threshold "
        "(2, 3 or 5 consecutive α points that C must stay top-1); y: `A|C|B` rate per eligible "
        "path. The three line styles are minimum-dominance-margin floors of 0, 0.02 and 0.05. The "
        "dotted vertical line marks the frozen default (persistence 3, margin > 0). The rate "
        "degrades smoothly and never collapses.",
    "natural_neighbor_comparison.png":
        "Do C-region activations sit where real activations sit? **Left:** distribution of cosine "
        "distance to the nearest of 2,000 held-out natural activations (x: cosine distance, lower = "
        "more natural; y: density) for A-region, C-region and B-region interpolation points and for "
        "natural contexts used as queries. **Right:** fraction of the 10 nearest natural neighbours "
        "whose own unpatched top-1 next token equals the query's own top-1 token, with 95% "
        "bootstrap intervals. C-region points are the furthest out and the least supported.",
    "continuation_stability.png":
        "Is the C region the same state throughout its run? x: the six inspected candidates, "
        "labelled with their C token and interpolation block; y: number of leading greedy-decoded "
        "tokens (out of 20) that are identical across continuations generated at the first, middle "
        "and last α of the C run. The dotted line at 1 is the trivial floor, since the first "
        "decoded token is C by construction. Two of six hold all 20 tokens; three collapse to the "
        "floor.",
    "depth_sweep.png":
        "Where in the network the effect lives (exploratory: same 1,000 pairs, so not independent "
        "evidence). **Left:** percentage of eligible paths with a persistent third top-1 token "
        "(solid, circles) and with a true sub-plateau, ρ < 0.5 (dashed, squares), against the "
        "interpolation block L (x, 0–30 of 36); error bars are 95% Wilson intervals and the "
        "hatched region marks the preregistered blocks 0–6. **Right:** median flatness ρ "
        "of the C window against L, preregistered blocks solid with circles and exploratory blocks "
        "dashed with squares; dashed rule at ρ = 1, dotted rule at ρ = 0.5. The trend "
        "turns over: the phenomenon is early-to-mid network.",
    "real_text_prevalence.png":
        "Real-language paths versus activation interpolation. **(A)** rate per eligible path under "
        "the symmetric rule (A, C and B runs each ≥ 3 grid points) for a persistent third "
        "token (hatched `//`) and for a true sub-plateau, ρ < 0.5 (hatched `\\\\`); error bars "
        "are 95% Wilson intervals. **(B)** transition width w(10→90) as a fraction of the "
        "path. **(C)** flatness ρ of the C window, clipped at 6. **(D)** the step k at which "
        "context B's prediction first becomes top-1 (0 = A's text, 32 = B's). **(E)** motion "
        "concentration κ, the share of total output motion Σ|Δd| carried by the "
        "sharpest 10% of steps; 0.1 (dashed rule) would mean a perfectly smooth ramp. **(F)** "
        "number of top-1 runs per path, clipped at 15. In B–F, solid = real text with random "
        "pairs, dashed = real text with final-token-matched pairs, dash-dot = activation-"
        "interpolation paths, dotted = ordinary (non-candidate) activation paths. Real-language "
        "third regions are far flatter (C) but the paths pass through many more predictions (F).",
    "real_text_examples.png":
        "What a real-language sub-plateau looks like — no patching anywhere; every marker is a "
        "real 32-token sequence run through the unmodified model. x: path position t = k/32, where "
        "k is the number of leading tokens already replaced by context B's; y: relative output "
        "distance d(t) on the final logits, 0 = the output looks like context A's prediction, 1 = "
        "like context B's. Dashed grey = the no-plateau diagonal d = t; shaded band = the detected "
        "C run; thin vertical lines = top-1 token changes; the A/C/B labels above each panel give "
        "the decoded tokens (␣ marks a leading space). **Top row:** the three highest-scoring "
        "qualifying paths from the random-pair bank R1. **Bottom row:** the same from the "
        "final-token-matched bank R2. Titles give the C run's step range and its flatness ρ.",
}

ALT = {
    "candidate_prevalence_by_layer.png": "A|C|B rate by block and by control condition",
    "top_candidate_probability_paths.png": "probability of A, C and B along six paths",
    "segment_width_margin_distribution.png": "C-segment width, margin and transition locations",
    "matthew_dt_frozen.png": "output-distance curves for the six pre-frozen inspection paths",
    "subplateau_dwell.png": "flatness of the C window against a matched control and the score",
    "matthew_dt_gallery.png": "output-distance curves for the six flattest sub-plateaus",
    "c_region_confidence.png": "top-1 probability and entropy in the C region vs the endpoints",
    "intermediate_token_census.png": "commonest intermediate and endpoint tokens",
    "threshold_sensitivity.png": "rate against the persistence and margin thresholds",
    "natural_neighbor_comparison.png": "nearest-natural-activation distance and neighbour agreement",
    "continuation_stability.png": "common greedy-prefix length across the C run",
    "depth_sweep.png": "third-token rate and flatness against interpolation block",
    "real_text_prevalence.png": "six-panel comparison of real-text paths with activation paths",
    "real_text_examples.png": "output-distance curves for six real-language A|C|B paths",
}

EMBED = re.compile(r"^!\[[^\]]*\]\((plots/([^)]+))\)\s*$")


def cite(out, n):
    """Append a '(Figure n)' reference to the last non-empty prose line before the figure."""
    for j in range(len(out) - 2, max(len(out) - 8, -1), -1):
        s = out[j].rstrip()
        if not s or s.startswith(("|", "!", "#", ">", "```")):
            continue
        if f"Figure {n}" in s or f"Figures {n}" in s:
            return
        out[j] = s[:-1] + f" (Figure {n}):" if s.endswith(":") else s + f" (Figure {n})"
        return


def process(path):
    lines = open(path).read().split("\n")
    out, n = [], 0
    i = 0
    while i < len(lines):
        m = EMBED.match(lines[i])
        if m:
            fname = m.group(2)
            out.append(f"![{ALT.get(fname, fname)}]({m.group(1)})")
        else:
            out.append(lines[i])
        if not m:
            i += 1
            continue
        fname = m.group(2)
        if fname not in CAPTIONS:
            raise SystemExit(f"{path}: no caption defined for {fname}")
        n += 1
        cite(out, n)
        i += 1
        # skip a blank line plus an existing caption paragraph, if present
        if i < len(lines) and lines[i].strip() == "":
            if i + 1 < len(lines) and lines[i + 1].startswith("**Figure "):
                i += 2
                while i < len(lines) and lines[i].strip() != "":
                    i += 1
        out.append("")
        out.append(f"**Figure {n}.** {CAPTIONS[fname]}")
    text = "\n".join(out)
    open(path, "w").write(text)
    print(f"{path}: {n} captions")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        process(p)
