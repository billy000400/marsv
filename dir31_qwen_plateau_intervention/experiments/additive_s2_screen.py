"""S4.2: screen on the tuning prompt only ("The capital city of Japan is").
Every candidate alone at beta = 0, 0.5, 1, 2, then within-layer groups: for each layer and beta > 0, the top 3 and
top 5 (where available) of the features that pass all three preservation checks at that beta, ranked by measured
answer effect. Each row carries the answer effect and all three checks. Reserved prompts are not run here.
Writes results/additive/tuning_base.json, screen_single.csv, screen_group.csv, screen_controls.csv and
tuning_best_controls.csv (three matched random controls for the best settings; descriptive, not used for selection)."""
import json
import os

import torch

import common as C
from additive_common import (BETAS, MIN_EFFECT, N_CTRL, OUT, TOL_PROB, TOL_READOUT_FRAC, AdditiveEdit, Bench, control_vector,
                             edit_vector, load_candidates, write_csv)

GROUP_SIZES = [3, 5]


def main():
    B = Bench()
    cands = load_candidates()
    ids, pos = B.capital("tune")
    base, gap = B.measure(ids, pos), B.gap("tune")
    # beta = 0 and hook removal must reproduce the unedited logits
    zero = B.measure(ids, pos, cands[0]["layer"], edit_vector(cands[0]["dec"][None], cands[0]["amp"][None], 0.0))
    again = B.measure(ids, pos)
    sanity = dict(beta0_max_abs_logit_diff=(zero["logits_last"] - base["logits_last"]).abs().max().item(),
                  hook_removed_max_abs_logit_diff=(again["logits_last"] - base["logits_last"]).abs().max().item())
    # the hook adds exactly the intended vector at the intended site (every candidate, beta = 1): the edited layer's
    # block output moves by the vector at the country token, earlier layers do not move, and the bridge token is
    # untouched up to and including the edited layer
    r0, r0b = C.run(B.model, ids, pos)["resid"][0], C.run(B.model, ids, pos, read_pos=pos + 1)["resid"][0]
    for k in ("site_max_abs_error", "upstream_max_abs_change", "bridge_upstream_max_abs_change"):
        sanity[k] = 0.0
    for c in cands:
        e = [AdditiveEdit(c["layer"], pos, edit_vector(c["dec"][None], c["amp"][None], 1.0))]
        d = C.run(B.model, ids, pos, edits=e)["resid"][0] - r0
        db = C.run(B.model, ids, pos, edits=e, read_pos=pos + 1)["resid"][0] - r0b
        sanity["site_max_abs_error"] = max(sanity["site_max_abs_error"], (d[c["layer"]] - e[0].vec).abs().max().item())
        sanity["upstream_max_abs_change"] = max(sanity["upstream_max_abs_change"], d[:c["layer"]].abs().max().item())
        sanity["bridge_upstream_max_abs_change"] = max(sanity["bridge_upstream_max_abs_change"], db[:c["layer"] + 1].abs().max().item())
    assert max(sanity.values()) < 1e-3, sanity
    info = dict(prompt=B.tok.decode(ids[0]), tokens=[B.tok.decode([i]) for i in ids[0].tolist()], country_position=pos,
                token_ids=B.tid, unedited={k: v for k, v in base.items() if k != "logits_last"},
                readout_gap_japan_minus_germany=gap, readout_tolerance=TOL_READOUT_FRAC * gap, prob_tolerance=TOL_PROB,
                min_answer_effect=MIN_EFFECT, sanity=sanity,
                controls=[dict(prompt=c["prompt"], relation=c["relation"], japan_position=c["pos"], answer=c["answer"],
                               p_answer_unedited=b["p_answer"], top1_unedited=b["top1"], readout26_unedited=b["readout26"])
                          for c, b in zip(B.controls, B.control_base)],
                excluded_first_token_controls=[c["prompt"] for c in B.excluded_controls])
    json.dump(info, open(os.path.join(OUT, "tuning_base.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    assert len(B.controls) == 4 and len(B.excluded_controls) == 3

    single, group, ctl_rows = [], [], []

    def trial(kind, layer, members, beta, ctrl_seed=None):
        amp = torch.stack([m["amp"] for m in members])
        vec = edit_vector(torch.stack([m["dec"] for m in members]), amp, beta)
        key = dict(kind=kind, layer=layer, features="+".join(str(m["feature"]) for m in members), n_features=len(members), beta=beta)
        if ctrl_seed is not None:
            vec, rand_rows = control_vector(layer, amp, beta, vec, ctrl_seed)
            key["random_decoder_rows"] = "+".join(map(str, rand_rows))
        row, crow = B.evaluate(ids, pos, base, gap, layer, vec)
        if ctrl_seed is None:
            ctl_rows.extend({**key, **c} for c in crow)
        return {**key, **row}

    for c in cands:
        for beta in BETAS:
            single.append(trial("single", c["layer"], [c], beta))
    assert all(abs(r["answer_effect"]) < 1e-4 for r in single if r["beta"] == 0)
    core = list(single[0])
    for beta in BETAS:  # rank of each feature among the candidates at the same strength, by measured answer effect
        for rank, r in enumerate(sorted((r for r in single if r["beta"] == beta), key=lambda r: -r["answer_effect"]), 1):
            r["rank_at_beta"] = rank if beta > 0 else ""
    by = {(c["layer"], c["feature"]): c for c in cands}
    for L in sorted({c["layer"] for c in cands}):
        for beta in BETAS[1:]:
            ok = sorted((r for r in single if r["layer"] == L and r["beta"] == beta and r["pass_all"]),
                        key=lambda r: -r["answer_effect"])
            for k in GROUP_SIZES:
                if len(ok) < k:
                    continue
                g = trial(f"top{k}", L, [by[(L, int(r["features"]))] for r in ok[:k]], beta)
                g["n_passing_in_layer"] = len(ok)
                g["member_single_effects"] = "+".join(f"{r['answer_effect']:.4f}" for r in ok[:k])
                g["all_members_positive"] = all(r["answer_effect"] > 0 for r in ok[:k])
                group.append(g)
    # Descriptive only, never used for selection: three matched random controls on the tuning prompt for the largest
    # answer effect overall and the largest one that passes all three checks (they may be the same setting).
    live = [r for r in single + group if r["beta"] > 0]
    targets = {id(r): r for r in (max(live, key=lambda r: r["answer_effect"]),
                                  max((r for r in live if r["pass_all"]), key=lambda r: r["answer_effect"]))}
    best_ctl = []
    for r in targets.values():
        members = [by[(r["layer"], int(f))] for f in r["features"].split("+")]
        best_ctl.append(dict(arm="edit", **{k: r[k] for k in core}, random_decoder_rows=""))
        for seed in range(1, N_CTRL + 1):
            best_ctl.append(dict(arm=f"control{seed}", **trial(r["kind"], r["layer"], members, r["beta"], seed)))
    write_csv("tuning_best_controls.csv", best_ctl)
    write_csv("screen_single.csv", single)
    write_csv("screen_controls.csv", ctl_rows)
    if group:
        write_csv("screen_group.csv", group)

    fmt = lambda r: (f"L{r['layer']:>2} {r['kind']:>6} b={r['beta']} n={r['n_features']} effect={r['answer_effect']:+.3f} "
                     f"dReadout={r['d_readout26']:+.4f} ({r['d_readout26_frac_of_gap']:+.1%} of gap) ctlFall={r['control_max_fall']:+.4f} "
                     f"top1kept={r['controls_top1_kept']}/4 isFall={r['is_fall']:+.4f} nextTok={r['top1_after_country']!r} "
                     f"pass R/C/I={int(r['pass_readout'])}{int(r['pass_controls'])}{int(r['pass_is'])} l2={r['edit_l2']:.2f} feats={r['features']}")
    print(f"\n{len(single)} single rows, {len(group)} group rows")
    print("\n== top 15 by answer effect (any preservation outcome)")
    for r in sorted(live, key=lambda r: -r["answer_effect"])[:15]:
        print(" ", fmt(r))
    print("\n== top 15 by answer effect among rows passing all three checks")
    for r in sorted((r for r in live if r["pass_all"]), key=lambda r: -r["answer_effect"])[:15]:
        print(" ", fmt(r))
    print("\n== matched random controls for the best settings (tuning prompt, descriptive)")
    for r in best_ctl:
        print(f"  {r['arm']:>8}", fmt(r))
    print("\n== all group rows")
    for r in group:
        print(" ", fmt(r))


if __name__ == "__main__":
    main()
