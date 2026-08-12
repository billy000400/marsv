"""Shared setup for dir23: same Japan->Germany embedding, five downstream readouts."""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

MODEL = "gpt2-large"

PREFIX = (
    "Country: France\n"
    " Capital: Paris\n"
    " Continent: Europe\n"
    " Currency: euro\n"
    " Language: French\n"
    " Type: country\n"
    "\n"
    "Country:"
)

ENDPOINT_A = " Japan"
ENDPOINT_B = " Germany"

# readout name -> (suffix, expected Japan-side answer, expected Germany-side answer)
READOUTS = [
    ("Capital",   "\n Capital:",   " Tokyo",    " Berlin"),
    ("Continent", "\n Continent:", " Asia",     " Europe"),
    ("Currency",  "\n Currency:",  " yen",      " euro"),
    ("Language",  "\n Language:",  " Japanese", " German"),
    ("Type",      "\n Type:",      " country",  " country"),
]

N_T = 101
ALPHAS = np.linspace(0.0, 1.0, N_T)

# green-free CVD palette (CLAUDE.md rule 13)
CVD = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]


def load():
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.225)
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m, dev


def slerp_lerp_norm(ea, eb, alphas):
    """Shortest-arc SLERP of direction; L2 norm interpolated linearly between endpoint norms."""
    na, nb = ea.norm(), eb.norm()
    u, v = ea / na, eb / nb
    cos = torch.clamp((u * v).sum(), -1.0, 1.0)
    om = torch.arccos(cos)
    a = torch.tensor(alphas, device=ea.device, dtype=ea.dtype).unsqueeze(1)
    if om.abs() < 1e-6:
        d = (1 - a) * u + a * v
        d = d / d.norm(dim=1, keepdim=True)
    else:
        d = (torch.sin((1 - a) * om) * u + torch.sin(a * om) * v) / torch.sin(om)
    return ((1 - a) * na + a * nb) * d, float(om), float(cos)


def jsd_bits(p, q):
    """Jensen-Shannon divergence in bits between two probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def h(x):
        x = x[x > 0]
        return float(-(x * np.log2(x)).sum())

    return h(m) - 0.5 * (h(p) + h(q))


def rel_dist(z, za, zb):
    """d(t) = |z-zA| / (|z-zA| + |z-zB|) for rows of z."""
    da = np.linalg.norm(z - za[None, :], axis=1)
    db = np.linalg.norm(z - zb[None, :], axis=1)
    return da / (da + db)


def crossings(alphas, d, level):
    """All linearly-interpolated crossings of `level`; returns list of t values."""
    out = []
    for i in range(len(d) - 1):
        lo, hi = d[i], d[i + 1]
        if (lo - level) == 0.0:
            out.append(float(alphas[i]))
        elif (lo - level) * (hi - level) < 0:
            f = (level - lo) / (hi - lo)
            out.append(float(alphas[i] + f * (alphas[i + 1] - alphas[i])))
    if d[-1] == level:
        out.append(float(alphas[-1]))
    return out


def transition_stats(alphas, d):
    """t10/t50/t90 (first crossing each), width, monotonicity and multiple-crossing flags."""
    res = {}
    for lv, key in ((0.1, "t10"), (0.5, "t50"), (0.9, "t90")):
        cs = crossings(alphas, d, lv)
        res[key] = cs[0] if cs else None
        res["n_cross_" + key] = len(cs)
    res["w"] = (res["t90"] - res["t10"]) if (res["t10"] is not None and res["t90"] is not None) else None
    diffs = np.diff(d)
    res["monotonic"] = bool((diffs >= -1e-12).all())
    res["max_backstep"] = float(-diffs.min()) if (diffs < 0).any() else 0.0
    return res
