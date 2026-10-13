"""S4.3b: the frozen edit on the two reserved prompts. One pass; nothing is selected or retuned here.
Arms: unedited; the frozen edit (results/additive/frozen_edit.json, read as is); three matched random controls (same
layer and position, same feature count and coefficients on random decoder rows, rescaled to the frozen vector's L2
norm, seeds 1-3). Per arm and prompt: answer effect, probabilities, top token, layer-26 readout change against the
gap of the matched capital template, the ' is' check at that prefix, the four Japanese controls, a greedy completion.
The request's three pilot criteria are judged separately on each reserved prompt.
Writes results/additive/reserved_eval.csv, reserved_controls.csv, pilot_verdict.json."""
import hashlib
import json
import os

import torch

import common as C
from additive_common import MIN_EFFECT, N_CTRL, OUT, TOL_PROB, TOL_READOUT_FRAC, AdditiveEdit, Bench, control_vector, edit_vector, load_candidates, write_csv

RESERVED = ["original", "city_of"]
N_NEW = 8
NO_EDIT = "No selective additive intervention found in this bounded candidate search."


def main():
    B = Bench()
    path = os.path.join(OUT, "frozen_edit.json")
    fz = json.load(open(path))
    assert fz["gate"] == "frozen_for_reserved_pilot", fz
    L, beta = fz["layer"], fz["beta"]
    by = {(c["layer"], c["feature"]): c for c in load_candidates()}
    members = [by[(L, f)] for f in fz["features"]]
    amp = torch.stack([m["amp"] for m in members])
    assert max(abs(a - b) for a, b in zip(amp.tolist(), fz["amplitudes"])) < 1e-6
    real = edit_vector(torch.stack([m["dec"] for m in members]), amp, beta)
    assert abs(real.norm().item() - fz["vector_l2"]) < 1e-3
    arms = [("unedited", None, ""), ("frozen edit", real, "+".join(map(str, fz["features"])))]
    for seed in range(1, N_CTRL + 1):
        vec, rand_rows = control_vector(L, amp, beta, real, seed)
        arms.append((f"random control {seed}", vec, "+".join(map(str, rand_rows))))

    rows, ctl_rows, site = [], {}, {}
    for template in RESERVED:
        ids, pos = B.capital(template)
        tap = C.d30.MLPTap(B.model)  # native MLP-output norm at the edit site of this prefix (unedited run)
        B.model(ids)
        site[template] = tap.y[L][0, pos].norm().item()
        tap.remove()
        base, gap = B.measure(ids, pos), B.gap(template)
        for arm, vec, dec_rows in arms:
            row, crow = B.evaluate(ids, pos, base, gap, L, vec)
            with C.edited(B.model, [] if vec is None else [AdditiveEdit(L, pos, vec)]):  # hook on for every pass, no KV cache
                g = B.model.generate(ids, do_sample=False, max_new_tokens=N_NEW, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=B.tok.eos_token_id, use_cache=False)
            rows.append(dict(template=template, prompt=B.tok.decode(ids[0]), country_position=pos, arm=arm, layer=L,
                             decoder_rows=dec_rows, beta=0.0 if vec is None else beta, readout_gap=gap,
                             readout_tolerance=TOL_READOUT_FRAC * gap, readout26_unedited=base["readout26"],
                             p_is_unedited=base["p_is"], **row, greedy_completion=B.tok.decode(g[0, ids.shape[1]:])))
            # the control prompts do not depend on the capital prompt, so they are stored once per arm
            assert arm not in ctl_rows or all(abs(a["p_answer"] - b["p_answer"]) < 1e-6 for a, b in zip(ctl_rows[arm], crow))
            ctl_rows.setdefault(arm, crow)
    write_csv("reserved_eval.csv", rows)
    write_csv("reserved_controls.csv", [dict(arm=arm, **c) for arm, cs in ctl_rows.items() for c in cs])

    per = {}
    for template in RESERVED:
        r = {x["arm"]: x for x in rows if x["template"] == template}
        e, ctrl = r["frozen edit"], [r[f"random control {s}"]["answer_effect"] for s in range(1, N_CTRL + 1)]
        per[template] = dict(
            prompt=e["prompt"], answer_effect=e["answer_effect"], control_answer_effects=ctrl,
            criterion1_at_least_plus1_toward_berlin=e["answer_effect"] >= MIN_EFFECT,
            criterion2_larger_than_every_random_control=e["answer_effect"] > max(ctrl),
            criterion3_all_preservation_checks_pass=e["pass_all"],
            pass_readout=e["pass_readout"], pass_controls=e["pass_controls"], pass_is=e["pass_is"],
            native_mlp_out_l2_site=site[template])
    crit = ["criterion1_at_least_plus1_toward_berlin", "criterion2_larger_than_every_random_control", "criterion3_all_preservation_checks_pass"]
    passes = all(per[t][c] for t in RESERVED for c in crit)
    verdict = dict(
        frozen_edit_sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), layer=L, features=fz["features"], beta=beta,
        vector_l2=real.norm().item(), thresholds=dict(min_answer_effect=MIN_EFFECT, readout_frac_of_gap=TOL_READOUT_FRAC, prob_fall=TOL_PROB),
        per_prompt=per, criteria_met_on_both_prompts={c: all(per[t][c] for t in RESERVED) for c in crit}, pilot_passes=passes,
        plateau_test="allowed: the pilot passed" if passes else "not run: the pilot did not pass",
        conclusion="pilot passed" if passes else NO_EDIT)
    json.dump(verdict, open(os.path.join(OUT, "pilot_verdict.json"), "w"), indent=1)
    for x in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k not in ("prompt", "decoder_rows", "layer", "beta")})
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
