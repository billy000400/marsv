"""S1: build the four answer groups, controls and backgrounds from the pinned LRE files,
run the unmodified-Qwen correctness filter, and save samples.jsonl + sample_counts.csv."""
import csv
import json
import os
import random
import re
import sys

import torch
import transformers

from common import LRE_DIR, LRE_REV, MODEL, MODEL_REV, OUT, SEED, TC_REPO, TC_REV, load_model, setup_torch

CITY_RELS = ["country_capital_city", "country_largest_city", "company_hq"]
COUNTRY_RELS = ["city_in_country", "landmark_in_country", "food_from_country"]
CONTEXT_RELS = ["country_currency", "country_language"]
GROUPS = {"Tokyo": CITY_RELS, "Japan": COUNTRY_RELS, "Berlin": CITY_RELS, "Germany": COUNTRY_RELS}
HELDOUT = {("country_capital_city", "Japan", "Tokyo"), ("country_capital_city", "Germany", "Berlin")}
TARGET_COUNTRIES = {"Japan", "Germany"}
# Explicit inclusion list for within-country city controls (object is a non-Tokyo Japanese city in company_hq).
JAPAN_CITY_OBJECTS = {"Kyoto"}
# company_hq objects that are countries or regions, not cities; the template asks for a city.
NON_CITY_OBJECTS = {"Bangladesh", "Belgium", "Egypt", "France", "Ireland", "Israel", "Italy", "Japan", "Malaysia",
                    "Netherlands", "Scotland", "Sweden", "California", "Colorado", "Hawaii", "Minnesota",
                    "Queensland", "Central"}
# Relations in which at least one primary group has discovery facts.
BACKGROUND_RELS = ["country_largest_city", "company_hq", "landmark_in_country", "food_from_country"]
N_BACKGROUND = 32
MAX_NEW_TOKENS = 16
# Declared before any feature result was inspected.
LEADING_ARTICLES = ["the"]
ALIASES = {"Yen": ["Japanese yen"], "New York City": ["New York"],
           "United States": ["United States of America", "USA", "U.S."]}


def norm(s):
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.lstrip(" \"'`*([{:-.,").strip()


def is_correct(completion, answer):
    c = norm(completion)
    for art in LEADING_ARTICLES:
        if c.startswith(art + " "):
            c = c[len(art) + 1:]
    for a in [answer] + ALIASES.get(answer, []):
        a = norm(a)
        if c.startswith(a) and not c[len(a):len(a) + 1].isalnum():
            return True
    return False


def build_rows():
    rows, anomalies = [], []
    rng_pick = {}
    for rel in CITY_RELS + COUNTRY_RELS + CONTEXT_RELS:
        d = json.load(open(os.path.join(LRE_DIR, rel + ".json")))
        templates = list(dict.fromkeys(d["prompt_templates"] + d["prompt_templates_zs"]))
        facts = list(dict.fromkeys((s["subject"], s["object"]) for s in d["samples"]))
        n_obj = {}
        for s, o in facts:
            n_obj.setdefault(s, set()).add(o)

        def clean_reason(s, o):
            if "\\u" in s or "\\u" in o or not s.strip() or not o.strip():
                return "malformed_source_text"
            if len(n_obj[s]) > 1:
                return "conflicting_labels"
            if rel == "company_hq" and o in NON_CITY_OBJECTS:
                return "non_city_object_in_city_relation"
            return ""

        role_of = {}
        for s, o in facts:
            for g, rels in GROUPS.items():
                if o == g and rel in rels:
                    role_of[(s, o)] = (g, "heldout" if (rel, s, o) in HELDOUT else "discovery")
            if rel in CONTEXT_RELS and s in TARGET_COUNTRIES:
                role_of[(s, o)] = (s + "-context", "context_control")
            if rel == "company_hq" and o in JAPAN_CITY_OBJECTS:
                role_of[(s, o)] = ("Japan-other-city", "city_control")
            if rel == "company_hq" and o in TARGET_COUNTRIES:
                role_of[(s, o)] = (o + "-anomaly", "anomaly_excluded")
        if rel in BACKGROUND_RELS:
            eligible = sorted((s, o) for s, o in facts if (s, o) not in role_of and not clean_reason(s, o)
                              and o not in GROUPS and s not in TARGET_COUNTRIES)
            rng_pick[rel] = len(eligible)
            for f in random.Random(SEED).sample(eligible, min(N_BACKGROUND, len(eligible))):
                role_of[f] = ("background", "background")
        for (s, o), (g, role) in role_of.items():
            for t in templates:
                prompt = t.format(s)
                reason = clean_reason(s, o)
                if not reason and re.search(r"\b" + re.escape(o.lower()) + r"\b", prompt.lower()):
                    reason = "answer_leaks_into_prompt"
                rows.append(dict(group=g, role=role, fact_id=f"{rel}|{s}|{o}", relation=rel, subject=s, answer=o,
                                 source=f"data/factual/{rel}.json", template=t, prompt=prompt,
                                 clean=not reason, clean_reason=reason))
    seen = set()
    rows = [r for r in rows if (k := (r["prompt"], r["answer"])) not in seen and not seen.add(k)]
    return rows, rng_pick


def main():
    setup_torch()
    rows, n_eligible = build_rows()
    tok, model = load_model()
    for r in rows:
        r.update(correct=False, completion=None, completion_token_ids=None, rejection_reason=r["clean_reason"],
                 last_token_index=None, last_token_text=None)
        if not r["clean"]:
            continue
        enc = tok(r["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.eos_token_id)
        new = out[0, enc.input_ids.shape[1]:].tolist()
        r["completion"] = tok.decode(new, skip_special_tokens=True)
        r["completion_token_ids"] = new
        r["correct"] = is_correct(r["completion"], r["answer"])
        r["rejection_reason"] = "" if r["correct"] else "completion_does_not_begin_with_answer"
        r["last_token_index"] = enc.input_ids.shape[1] - 1
        r["last_token_text"] = tok.decode(enc.input_ids[0, -1:])
    with open(os.path.join(OUT, "samples.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Counts: facts and prompts kept separate, per group x relation and per group total.
    def count(sel):
        return len({r["fact_id"] for r in sel}), len(sel)

    out = []
    for g in dict.fromkeys(r["group"] for r in rows):
        gr = [r for r in rows if r["group"] == g]
        for rel in ["ALL"] + list(dict.fromkeys(r["relation"] for r in gr)):
            rr = [r for r in gr if rel in ("ALL", r["relation"])]
            stages = {"raw": rr, "clean": [r for r in rr if r["clean"]],
                      "model_correct": [r for r in rr if r["correct"]],
                      "discovery": [r for r in rr if r["correct"] and r["role"] == "discovery"],
                      "heldout": [r for r in rr if r["correct"] and r["role"] == "heldout"]}
            row = dict(group=g, relation=rel)
            for name, sel in stages.items():
                row[name + "_facts"], row[name + "_prompts"] = count(sel)
            out.append(row)
    with open(os.path.join(OUT, "sample_counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for row in out:
        print(row)

    cfg = dict(model=MODEL, model_revision=MODEL_REV, transcoder_repo=TC_REPO, transcoder_revision=TC_REV,
               lre_repo="evandez/relations", lre_revision=LRE_REV, seed=SEED, model_dtype="float32",
               device=torch.cuda.get_device_name(0), python=sys.version, torch=torch.__version__,
               transformers=transformers.__version__,
               tokenizer=dict(call="tok(prompt, return_tensors='pt')", add_special_tokens="default (Qwen adds none)",
                              chat_template=False, padding=False),
               decoding=dict(do_sample=False, max_new_tokens=MAX_NEW_TOKENS, batch_size=1),
               correctness=dict(rule="normalised completion begins with answer or alias, followed by a non-alphanumeric "
                                     "character or end of text", leading_articles=LEADING_ARTICLES, aliases=ALIASES),
               groups=GROUPS, heldout=sorted(map(list, HELDOUT)), context_relations=CONTEXT_RELS,
               japan_city_control_objects=sorted(JAPAN_CITY_OBJECTS), non_city_objects=sorted(NON_CITY_OBJECTS),
               background=dict(n_per_relation=N_BACKGROUND, eligible_facts=n_eligible,
                               excluded="target answers, subjects Japan/Germany, control facts, unclean facts"))
    json.dump(cfg, open(os.path.join(OUT, "run_config.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
