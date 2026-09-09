"""S0 - count matched examples for the two candidate features and dump readable examples.

No model is loaded. Positions are indices into the tinyshakespeare corpus; a "context" is the
128-character window ending at that position (the model's block size). Direction-building examples
come from the first 90% of the corpus (dir13's training split), behavioural-test examples from the
last 10% (dir13's held-out split), so no example used to build a direction is used to test it.
"""
import os, json, hashlib, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CORPUS = "/tmp/tinyshakespeare.txt"
CORPUS_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CTX = 128


def load():
    raw = open(CORPUS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA
    return raw.decode("utf-8")


def line_spans(text):
    """For every index, the (start, end) of its line and whether the line is an all-caps label."""
    starts = [0]
    for i, c in enumerate(text):
        if c == "\n":
            starts.append(i + 1)
    spans = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 1 if k + 1 < len(starts) else len(text)
        body = text[s:e]
        is_label = body.endswith(":") and len(body) > 1
        allcaps = is_label and all(c.isupper() or c == " " for c in body[:-1]) and any(c.isalpha() for c in body)
        spans.append((s, e, is_label, allcaps))
    return spans


def index_maps(text, spans):
    """Per-index line id array."""
    lid = [0] * len(text)
    for k, (s, e, _, _) in enumerate(spans):
        for i in range(s, min(e + 1, len(text))):
            lid[i] = k
    return lid


def collect(text, lo, hi, spans, lid):
    """Candidate positions with a full 128-char context inside [lo, hi)."""
    f1p, f1n, f2p, f2n = [], [], [], []
    for i in range(lo + CTX, hi - 1):
        c = text[i]
        k = lid[i]
        s, e, is_label, allcaps = spans[k]
        if c.isupper() and c.isalpha():
            if allcaps and i < e - 1:              # inside the label body, before the colon
                f1p.append(i)
            elif not is_label:                     # ordinary (non-label) line
                f1n.append(i)
        if c.isalpha() and not is_label:
            (f2p if text[i + 1].isalpha() else f2n).append(i)
    return {"speaker_pos": f1p, "speaker_neg": f1n, "wordcont_pos": f2p, "wordcont_neg": f2n}


def matched_counts(pos, neg, text):
    """How many matched pairs are available: per final character, min(#pos, #neg)."""
    cp = collections.Counter(text[i] for i in pos)
    cn = collections.Counter(text[i] for i in neg)
    per = {c: min(cp[c], cn[c]) for c in set(cp) | set(cn)}
    per = {c: v for c, v in sorted(per.items(), key=lambda kv: -kv[1]) if v > 0}
    return int(sum(per.values())), per, dict(cp), dict(cn)


def show(text, i, before=48):
    left = text[max(0, i - before):i].replace("\n", "\\n")
    return f"...{left}[{text[i]}]  -> next {text[i+1]!r}"


def main():
    text = load()
    spans = line_spans(text)
    lid = index_maps(text, spans)
    n = int(0.9 * len(text))
    out, lines = {}, []
    for split, lo, hi in [("train", 0, n), ("heldout", n, len(text))]:
        cand = collect(text, lo, hi, spans, lid)
        out[split] = {}
        for feat, p, q in [("speaker_label", "speaker_pos", "speaker_neg"),
                           ("word_continuation", "wordcont_pos", "wordcont_neg")]:
            tot, per, cp, cn = matched_counts(cand[p], cand[q], text)
            out[split][feat] = {"n_positive": len(cand[p]), "n_negative": len(cand[q]),
                                "matched_pairs_available": tot,
                                "matched_pairs_per_final_char": per}
            lines.append(f"[{split}] {feat}: positives={len(cand[p])} negatives={len(cand[q])} "
                         f"matched pairs available={tot}")
        if split == "train":
            for feat, p, q in [("speaker_label", "speaker_pos", "speaker_neg"),
                               ("word_continuation", "wordcont_pos", "wordcont_neg")]:
                lines.append("")
                lines.append(f"=== {feat} (train split) - 10 positive examples ===")
                lines += [show(text, i) for i in cand[p][:2000:199][:10]]
                lines.append(f"=== {feat} (train split) - 10 negative examples ===")
                lines += [show(text, i) for i in cand[q][:2000:199][:10]]
    json.dump(out, open(os.path.join(RES, "feature_counts.json"), "w"), indent=2)
    open(os.path.join(RES, "feature_examples.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
