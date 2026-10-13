"""Shared helpers for stage S4 (operator addendum): a strictly additive edit at the country token.
    y_new = y_native + beta * sum_i(reference_amplitude_i * decoder_i)
Reuses dir31's common.py unchanged (model, hooks, run). Nothing here subtracts activations or replaces the MLP."""
import csv
import json
import os

import numpy as np
import torch

import common as C

OUT = os.path.join(C.OUT, "additive")
PLOTS = os.path.join(C.PLOTS, "additive")
D30_OUT = C.d30.OUT
BETAS = [0.0, 0.5, 1.0, 2.0]
LAYERS = range(1, 27)        # feedback: layers 1-26
GERMANY_CUTOFF = "top50"     # broadest dir30 Germany set (JOURNAL assumption 1)
READ_LAYER = 26
TOL_READOUT_FRAC = 0.10      # of the unedited Japan-Germany readout gap, matched capital template
TOL_PROB = 0.05              # absolute probability
MIN_EFFECT = 1.0             # logits toward Berlin
N_CTRL = 3
CHECKS = ["pass_readout", "pass_controls", "pass_is"]


class AdditiveEdit:
    """Adds one fixed vector to a layer's native MLP output at one position. Same interface as common.Edit, so
    common.run / common.edited apply it; unlike common.Edit the vector does not depend on the current activations."""

    def __init__(self, layer, pos, vec):
        self.layer, self.pos, self.vec, self.norm = layer, pos, vec, vec.norm()

    def delta(self, x):
        return self.vec


def edit_vector(dec, amp, beta):
    """dec [k, d] decoder rows, amp [k] reference amplitudes."""
    return beta * (amp @ dec)


def control_vector(layer, amp, beta, real, seed):
    """dir31's control: the same coefficients on random decoder rows of the same dictionary, rescaled to the real
    vector's L2 norm. Rows come from dir31's stored 64 random rows of this layer."""
    sl = torch.load(os.path.join(C.SLICES, f"layer_{layer}.pt"))
    g = torch.Generator().manual_seed(1000 * seed + layer)
    pick = torch.randperm(sl["rand_dec"].shape[0], generator=g)[:len(amp)]
    c = beta * (amp @ sl["rand_dec"][pick].to(C.DEV))
    return c * (real.norm() / c.norm()), sl["rand_ids"][pick].tolist()


def load_candidates():
    z = np.load(os.path.join(OUT, "candidates.npz"))
    return [dict(layer=int(l), feature=int(f), amp=torch.tensor(a, device=C.DEV), dec=torch.tensor(d, device=C.DEV))
            for l, f, a, d in zip(z["layer"], z["feature"], z["amplitude"], z["W_dec"])]


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


class Bench:
    """Model plus everything the three preservation checks need."""

    def __init__(self):
        self.tok, self.model = C.setup()
        tid = {w: self.tok(" " + w).input_ids for w in ["Tokyo", "Berlin", "is", "Japan", "Germany"]}
        assert all(len(v) == 1 for v in tid.values()), tid
        self.tid = {k: v[0] for k, v in tid.items()}
        pr = np.load(os.path.join(C.OUT, "probe.npz"))  # dir31's frozen country readout
        w, mid, scale = (torch.tensor(pr[k], device=C.DEV) for k in ("w", "mid", "scale"))
        self.probe = lambda h: ((h - mid) * w).sum(-1) / scale  # [..., L, d] -> [..., L]; +1 Japan, -1 Germany
        # dir30's model-correct Japanese currency / language prompts. Those starting with the country word are the
        # first-token controls dir31 found invalid; they are listed but never used.
        self.controls, self.excluded_controls = [], []
        for smp in map(json.loads, open(os.path.join(D30_OUT, "samples.jsonl"))):
            if smp["role"] != "context_control" or not smp["correct"] or smp["subject"] != "Japan":
                continue
            enc = self.tok(smp["prompt"], return_offsets_mapping=True)
            start = smp["prompt"].index("Japan")
            pos = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < start + len("Japan")][-1]
            ans = self.tok(" " + (smp["answer"].lower() if smp["answer"] == "Yen" else smp["answer"])).input_ids
            assert len(ans) == 1, ans
            c = dict(prompt=smp["prompt"], relation=smp["relation"], pos=pos, answer_id=ans[0],
                     answer=self.tok.decode(ans), ids=C.ids_of(self.tok, smp["prompt"]))
            (self.excluded_controls if pos == 0 else self.controls).append(c)
        for c in self.controls:
            assert c["ids"][0, c["pos"]].item() == self.tid["Japan"], c
        self.control_base = self.measure_controls()
        for c, b in zip(self.controls, self.control_base):  # each kept control is answered correctly unedited
            assert b["top1_id"] == c["answer_id"], (c["prompt"], b)

    def capital(self, template, country="Japan"):
        prefix, bridge = C.TEMPLATES[template]
        pos = self.tok(prefix.format(country), return_tensors="pt").input_ids.shape[1] - 1
        ids = C.ids_of(self.tok, prefix.format(country) + bridge)
        assert ids[0, pos].item() == self.tid[country] and ids[0, pos + 1].item() == self.tid["is"]
        return ids, pos

    def measure(self, ids, pos, layer=None, vec=None):
        """Capital prompt: answer logits after the bridge, next-token distribution at the country token, readout."""
        r = C.run(self.model, ids, pos, edits=[] if vec is None else [AdditiveEdit(layer, pos, vec)])
        ll, lp, lpp = r["logits_last"][0], r["logits_last"][0].log_softmax(-1), r["logits_pos"][0].log_softmax(-1)
        t = self.tid
        return dict(berlin_minus_tokyo=(ll[t["Berlin"]] - ll[t["Tokyo"]]).item(), logp_berlin=lp[t["Berlin"]].item(),
                    p_tokyo=lp[t["Tokyo"]].exp().item(), p_berlin=lp[t["Berlin"]].exp().item(),
                    top1=self.tok.decode(lp.argmax().item()), p_is=lpp[t["is"]].exp().item(),
                    top1_after_country_id=lpp.argmax().item(), top1_after_country=self.tok.decode(lpp.argmax().item()),
                    readout26=self.probe(r["resid"][0])[READ_LAYER].item(), logits_last=ll)

    def measure_controls(self, layer=None, vec=None):
        """The four mid-sentence Japanese controls with the same vector at their ' Japan' token."""
        out = []
        for c in self.controls:
            r = C.run(self.model, c["ids"], c["pos"], edits=[] if vec is None else [AdditiveEdit(layer, c["pos"], vec)])
            lp = r["logits_last"][0].log_softmax(-1)
            out.append(dict(p_answer=lp[c["answer_id"]].exp().item(), top1_id=lp.argmax().item(),
                            top1=self.tok.decode(lp.argmax().item()), readout26=self.probe(r["resid"][0])[READ_LAYER].item()))
        return out

    def gap(self, template):
        """Unedited Japan-minus-Germany layer-26 readout at the country token of this capital template."""
        return self.measure(*self.capital(template, "Japan"))["readout26"] - self.measure(*self.capital(template, "Germany"))["readout26"]

    def evaluate(self, ids, pos, base, gap, layer, vec):
        """One edit on one capital prompt: answer effect plus the three fixed pilot tolerances.
        Returns (summary row, per-control rows)."""
        m, ctl = self.measure(ids, pos, layer, vec), self.measure_controls(layer, vec)
        falls = [b["p_answer"] - c["p_answer"] for c, b in zip(ctl, self.control_base)]
        kept = [c["top1_id"] == b["top1_id"] for c, b in zip(ctl, self.control_base)]
        d_read, is_fall = m["readout26"] - base["readout26"], base["p_is"] - m["p_is"]
        row = dict(
            answer_effect=m["berlin_minus_tokyo"] - base["berlin_minus_tokyo"], berlin_minus_tokyo=m["berlin_minus_tokyo"],
            d_logp_berlin=m["logp_berlin"] - base["logp_berlin"], p_tokyo=m["p_tokyo"], p_berlin=m["p_berlin"], top1=m["top1"],
            readout26=m["readout26"], d_readout26=d_read, d_readout26_frac_of_gap=d_read / gap,
            pass_readout=abs(d_read) <= TOL_READOUT_FRAC * gap,
            controls_top1_kept=sum(kept), control_max_fall=max(falls),
            pass_controls=all(kept) and max(falls) <= TOL_PROB,
            p_is=m["p_is"], is_fall=is_fall, top1_after_country=m["top1_after_country"],
            pass_is=m["top1_after_country_id"] == self.tid["is"] and is_fall <= TOL_PROB,
            edit_l2=0.0 if vec is None else vec.norm().item())
        row["pass_all"] = all(row[k] for k in CHECKS)
        crow = [dict(prompt=c["prompt"], answer=c["answer"], p_answer_unedited=b["p_answer"], p_answer=x["p_answer"],
                     fall=f, top1=x["top1"], top1_kept=k, readout26_unedited=b["readout26"], readout26=x["readout26"])
                for c, b, x, f, k in zip(self.controls, self.control_base, ctl, falls, kept)]
        return row, crow
