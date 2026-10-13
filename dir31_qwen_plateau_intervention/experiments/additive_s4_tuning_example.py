"""S4 descriptive example on the TUNING prompt only. Nothing is selected here and no reserved prompt is run.
The feedback asks to "show probabilities, readout changes, and greedy completions" for the frozen edit on the
reserved prompts; no setting qualified, so that evaluation was not run. This script shows the same quantities on
the tuning prompt for the unedited model and for the setting with the largest answer effect in the screen
(read from results/additive/tuning_summary.json). Writes results/additive/tuning_example.csv."""
import json
import os

import torch

import common as C
from additive_common import OUT, AdditiveEdit, Bench, edit_vector, load_candidates, write_csv

N_NEW = 8


def main():
    B = Bench()
    best = json.load(open(os.path.join(OUT, "tuning_summary.json")))["best_answer_effect_overall"]
    by = {(c["layer"], c["feature"]): c for c in load_candidates()}
    members = [by[(best["layer"], int(f))] for f in best["features"].split("+")]
    vec = edit_vector(torch.stack([m["dec"] for m in members]), torch.stack([m["amp"] for m in members]), best["beta"])
    ids, pos = B.capital("tune")
    base, gap = B.measure(ids, pos), B.gap("tune")
    rows = []
    for arm, edits in [("unedited", []), ("largest answer effect in the screen", [AdditiveEdit(best["layer"], pos, vec)])]:
        m = B.measure(ids, pos, best["layer"], vec) if edits else base
        with C.edited(B.model, edits):  # the hook stays on for every forward pass of the greedy loop (no KV cache)
            g = B.model.generate(ids, do_sample=False, max_new_tokens=N_NEW, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=B.tok.eos_token_id, use_cache=False)
        rows.append(dict(
            arm=arm, prompt=B.tok.decode(ids[0]), layer=best["layer"] if edits else "", features=best["features"] if edits else "",
            beta=best["beta"] if edits else 0.0, edit_l2=vec.norm().item() if edits else 0.0,
            berlin_minus_tokyo=m["berlin_minus_tokyo"], answer_effect=m["berlin_minus_tokyo"] - base["berlin_minus_tokyo"],
            p_tokyo=m["p_tokyo"], p_berlin=m["p_berlin"], top1=m["top1"], readout26=m["readout26"],
            d_readout26=m["readout26"] - base["readout26"], d_readout26_frac_of_gap=(m["readout26"] - base["readout26"]) / gap,
            p_is=m["p_is"], top1_after_country=m["top1_after_country"], greedy_completion=B.tok.decode(g[0, ids.shape[1]:])))
    assert abs(rows[1]["answer_effect"] - best["answer_effect"]) < 1e-4, (rows[1]["answer_effect"], best["answer_effect"])
    write_csv("tuning_example.csv", rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
