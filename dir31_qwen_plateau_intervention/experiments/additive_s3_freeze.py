"""S4.3a: selection on the tuning screen. Reads only results/additive/screen_*.csv, candidates.csv and dir31's site
norms; runs no model and no reserved prompt. Writes three files.

preregistered_gate.json: outcome of the rule declared in JOURNAL.md before the screen. Qualifying = all three
preservation checks pass AND answer effect >= +1 on the tuning prompt; smallest = fewest features, then smaller
beta, then larger answer effect.

frozen_edit.json: the edit frozen for the reserved-prompt pilot. If the pre-registered rule selects a setting, that
one. If none reaches +1 on tuning, the operator's resolution of "qualifying" applies (set after the tuning screen and
before any reserved-prompt run): the setting with the largest answer effect among those that pass all three checks.
That is the largest passing edit, not the smallest. An existing frozen_edit.json is never rewritten: the script only
checks that it still matches.

tuning_summary.json: pass counts and best settings of the screen."""
import csv
import json
import os

import common as C
from additive_common import CHECKS, MIN_EFFECT, OUT, read_csv

PREREG_RULE = ("pass all three checks and answer effect >= +1 on the tuning prompt; "
               "smallest = fewest features, then smaller beta, then larger answer effect")
RULING_RULE = "largest answer effect on the tuning prompt among settings that pass all three preservation checks"
COST = ["answer_effect", "d_logp_berlin", "p_tokyo", "p_berlin", "top1", "d_readout26", "d_readout26_frac_of_gap", "pass_readout",
        "controls_top1_kept", "control_max_fall", "pass_controls", "p_is", "is_fall", "top1_after_country", "pass_is", "pass_all", "edit_l2"]


def parse(r):
    out = dict(r)
    for k, v in r.items():
        if v in ("True", "False"):
            out[k] = v == "True"
        elif v != "" and k not in ("kind", "features", "top1", "top1_after_country", "member_single_effects"):
            out[k] = float(v)
    for k in ("layer", "n_features", "controls_top1_kept"):
        out[k] = int(out[k])
    return out


def main():
    rows = [parse(r) for r in read_csv("screen_single.csv")]
    if os.path.exists(os.path.join(OUT, "screen_group.csv")):
        rows += [parse(r) for r in read_csv("screen_group.csv")]
    rows = [r for r in rows if r["beta"] > 0]
    brief = lambda r: None if r is None else {k: r[k] for k in ["kind", "layer", "features", "n_features", "beta"] + COST}
    best = lambda rs: max(rs, key=lambda r: r["answer_effect"]) if rs else None
    passing = [r for r in rows if r["pass_all"]]
    qualifying = sorted((r for r in passing if r["answer_effect"] >= MIN_EFFECT),
                        key=lambda r: (r["n_features"], r["beta"], -r["answer_effect"]))
    summary = dict(
        n_settings=len(rows), n_single=sum(r["kind"] == "single" for r in rows), n_group=sum(r["kind"] != "single" for r in rows),
        n_pass_each={k: sum(r[k] for r in rows) for k in CHECKS}, n_pass_all=len(passing),
        n_effect_at_least_1=sum(r["answer_effect"] >= MIN_EFFECT for r in rows), n_qualifying=len(qualifying),
        best_answer_effect_overall=brief(best(rows)), best_answer_effect_single=brief(best([r for r in rows if r["kind"] == "single"])),
        best_preservation_passing=brief(best(passing)),
        best_per_beta={str(b): dict(overall=brief(best([r for r in rows if r["beta"] == b])),
                                    passing=brief(best([r for r in passing if r["beta"] == b]))) for b in (0.5, 1.0, 2.0)},
        qualifying=[brief(r) for r in qualifying])
    json.dump(summary, open(os.path.join(OUT, "tuning_summary.json"), "w"), indent=1)

    gate = dict(rule=PREREG_RULE, declared="JOURNAL.md, entry of 2026-10-11 written before the screen", n_settings=len(rows),
                n_pass_all=len(passing), n_effect_at_least_1=summary["n_effect_at_least_1"], n_qualifying=len(qualifying),
                outcome="qualified" if qualifying else "none_qualified")
    json.dump(gate, open(os.path.join(OUT, "preregistered_gate.json"), "w"), indent=1)
    print(json.dumps(gate, indent=1))

    r = qualifying[0] if qualifying else best(passing)
    cand = {(int(c["layer"]), int(c["feature"])): c for c in read_csv("candidates.csv")}
    feats = [int(f) for f in r["features"].split("+")]
    d31 = {(x["template"], int(x["layer"])): float(x["mlp_out_l2"]) for x in csv.DictReader(open(os.path.join(C.OUT, "s2_slices.csv")))
           if x["country"] == "Japan" and x["site"] == "country"}
    frozen = dict(
        gate="frozen_for_reserved_pilot",
        selection_rule=PREREG_RULE if qualifying else RULING_RULE,
        selection_basis="pre-registered rule" if qualifying else (
            "operator's resolution of \"qualifying\" (2026-10-11): no setting met the pre-registered bar, so the largest "
            "preservation-passing setting is frozen; set after the tuning screen and before any reserved-prompt run; "
            "this is the largest passing edit, not the smallest"),
        preregistered_gate=dict(rule=PREREG_RULE, n_qualifying=len(qualifying), met=bool(qualifying)),
        site="country token, MLP output", layer=r["layer"], features=feats, kind=r["kind"],
        amplitudes=[float(cand[(r["layer"], f)]["amplitude"]) for f in feats], beta=r["beta"], vector_l2=r["edit_l2"],
        native_mlp_out_l2_tune_site=float(cand[(r["layer"], feats[0])]["native_mlp_out_l2_tune_site"]),
        native_mlp_out_l2_reserved_site_dir31=d31[("original", r["layer"])],
        formula="y_new = y_native + beta * sum_i(amplitude_i * W_dec[i]); no b_dec, no encoder term, nothing subtracted",
        tuning_result=brief(r), selected_from="tuning prompt only (screen_single.csv, screen_group.csv)")
    path = os.path.join(OUT, "frozen_edit.json")
    if os.path.exists(path):
        assert json.load(open(path)) == json.loads(json.dumps(frozen)), "frozen_edit.json exists and differs; it is never rewritten"
        print("frozen_edit.json already exists and matches")
    else:
        json.dump(frozen, open(path, "w"), indent=1)
    print(json.dumps(frozen, indent=1))


if __name__ == "__main__":
    main()
