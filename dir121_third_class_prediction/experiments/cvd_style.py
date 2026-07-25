"""Colour-vision-deficiency-safe plotting helpers (CLAUDE.md rule 13).

The operator of this project has red-green colour deficiency, so no figure may
carry meaning through a red-vs-green contrast and no series may be identified by
colour alone. Everything here is green-free and pairs every colour with a second
channel (linestyle / marker / hatch).
"""
import matplotlib.pyplot as plt

CVD = ["#0072B2",   # blue
       "#D55E00",   # vermillion
       "#CC79A7",   # reddish purple
       "#56B4E9",   # sky blue
       "#E69F00"]   # orange

LINESTYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D']
HATCHES = ['//', '\\\\', 'xx', '..']

# Stage-1 category encoding, shared by the transition matrix and the curve grid.
CAT_COLORS = {0: '#e8e8e8', 1: CVD[0], 2: CVD[1], 3: CVD[2]}
CAT_HATCH = {0: '', 1: '//', 2: '\\\\', 3: 'xx'}
CAT_LS = {0: '-', 1: '-', 2: '--', 3: '-.'}
CAT_NAMES = {0: 'neither', 1: 'stable third-class only',
             2: 'stable sub-plateau only', 3: 'both'}


def use_cvd():
    """Make the default property cycle green-free."""
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CVD)


def digit_color(c):
    """Colour for digit 0-9 on a sequential, CVD-safe ramp.

    Ten nominal series exceed the five-hue palette, so digit identity rides on
    a monotone `cividis` ramp (light->dark reads correctly in grayscale) and is
    always ALSO printed as a text label or named in the legend.
    """
    return plt.cm.cividis(c / 9.0)


def digit_text_color(c):
    """Readable text colour on top of digit_color(c)."""
    return 'w' if c <= 5 else 'k'
