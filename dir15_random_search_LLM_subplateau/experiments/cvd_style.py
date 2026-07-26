"""Colour-vision-deficiency-safe plotting constants (CLAUDE.md rule 13).

The operator has red-green colour deficiency, so no figure may carry meaning through a
red-vs-green contrast and no series may be identified by colour alone. This palette is
green-free; every user pairs a colour with a second channel (linestyle / marker / hatch).
"""
import matplotlib.pyplot as plt

CVD = ["#0072B2",   # blue
       "#D55E00",   # vermillion
       "#CC79A7",   # reddish purple
       "#56B4E9",   # sky blue
       "#E69F00"]   # orange

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]

REF_DIAG = dict(color="0.45", ls="--", lw=1.4)    # straight-line / no-plateau reference
REF_RULE = dict(color="k", ls=":", lw=1.6)        # frozen plateau-rule threshold


def use_cvd():
    """Make the default property cycle green-free."""
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
