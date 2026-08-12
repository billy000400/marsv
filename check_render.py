"""Check that a Markdown deliverable actually renders on GitHub -- and reads like a paper.

Six checks, each of which has caught a real project bug:

1. KaTeX compile of every display (```math fence) equation.
2. KaTeX compile of every inline $...$ expression *after* simulating GitHub's
   backslash-before-punctuation stripping (CLAUDE.md rule 8b).
3. Macros GitHub's markdown math renderer rejects outright ("The following macros
   are not allowed: ..."), e.g. \\operatorname -- use \\mathrm / \\text instead.
4. Display-math placement via the GitHub markdown API (rule 8a) + figure embeds (rule 12).
5. Every table is preceded by a prose sentence that states the claim (rule 9a) -- a
   table the reader has to interpret unaided is an unfinished result.
6. Mechanical contrast constructions ("we do X rather than Y") stay rare (rule 9d).

Usage: python check_render.py DIR/REPORT.md DIR/RESULTS.md
Exit code 1 if any check fails.
"""

import json
import re
import subprocess
import sys
import urllib.request

HERE = __import__("pathlib").Path(__file__).resolve().parent

# Confirmed rejected by GitHub's renderer (operator screenshot, 2026-07-26; also
# github/community#55368). The definition/HTML macros are blocked for security.
BAD_MACROS = [
    "operatorname",
    "def", "gdef", "edef", "xdef", "let",
    "newcommand", "renewcommand", "providecommand",
    "DeclareMathOperator", "includegraphics", "href", "url",
    "htmlClass", "htmlId", "htmlStyle", "htmlData",
]

FENCE_RE = re.compile(r"```math\n(.*?)```", re.S)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# The rule-9d tic is a *self-description* built on contrast ("we do X rather than Y"),
# not any use of the words -- "explained by A rather than B" is ordinary comparison.
CONTRAST_RE = re.compile(r"\b(rather than|instead of|not just|as opposed to)\b", re.I)
SELF_RE = re.compile(r"\b(we|our|ours|this (?:paper|report|method|work))\b", re.I)
UNLIKE_RE = re.compile(r"\bunlike (?:prior|existing|previous|other|standard)\b", re.I)
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
CONTRAST_BUDGET = 2
INLINE_RE = re.compile(r"(?<![$\\])\$(?!\$)([^$\n]+?)\$(?!\$)")
# GitHub strips a backslash before ASCII punctuation inside inline math.
GH_STRIP_RE = re.compile(r"\\([_{}|,;!:%#&~^$\\'\"()\[\]<>*+=/?@`.-])")


def strip_fences(text):
    """Blank out fenced blocks so inline scanning ignores them."""
    return re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)


def line_of(text, idx):
    return text.count("\n", 0, idx) + 1


def katex(items):
    cmd = ["node", str(HERE / "katex_compile.js")]
    p = subprocess.run(cmd, input=json.dumps(items), capture_output=True, text=True)
    if p.returncode != 0:  # fresh pod: no katex installed yet
        subprocess.run(["npm", "install", "--silent", "--prefix", "/tmp/katexcheck", "katex"],
                       check=True, capture_output=True)
        p = subprocess.run(cmd, input=json.dumps(items), capture_output=True, text=True,
                           check=True)
    return json.loads(p.stdout)


def is_prose(line):
    """A line that can carry a claim: not a heading, table row, image, fence or rule."""
    s = line.strip()
    return bool(s) and not s.startswith(("#", "|", "![", "```", "~~~", "---", "<"))


def unmotivated_tables(outside):
    """Rule 9a: a real sentence must sit above each table, not just a heading.

    The unit is the whole paragraph above the table, since prose wraps -- a
    paragraph ending in a short line like "measurement error:" still counts.
    """
    lines = outside.split("\n")
    bad = []
    for i, line in enumerate(lines[:-1]):
        if not (TABLE_ROW_RE.match(line) and TABLE_SEP_RE.match(lines[i + 1])):
            continue
        if i and TABLE_ROW_RE.match(lines[i - 1]):
            continue  # mid-table, not the header row
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        para = []
        while j >= 0 and is_prose(lines[j]):
            para.append(lines[j])
            j -= 1
        if len(" ".join(para).split()) < 12:
            bad.append(i + 1)
    return bad


def contrast_tics(outside):
    """Rule 9d: contrast constructions used to describe our own method."""
    hits = []
    for m in SENTENCE_RE.finditer(outside):
        s = m.group(0)
        if UNLIKE_RE.search(s) or (CONTRAST_RE.search(s) and SELF_RE.search(s)):
            hits.append((line_of(outside, m.start()), s.strip()))
    return hits


def github_render(text):
    req = urllib.request.Request(
        "https://api.github.com/markdown",
        data=json.dumps({"mode": "gfm", "text": text}).encode(),
        headers={"Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def check(path):
    text = open(path).read()
    outside = strip_fences(text)
    fails = []

    fences = [(m.start(), m.group(1)) for m in FENCE_RE.finditer(text)]
    inlines = [(m.start(), m.group(1)) for m in INLINE_RE.finditer(outside)]

    items = [{"id": f"display L{line_of(text, s)}", "tex": t, "display": True}
             for s, t in fences]
    # Inline: compile what GitHub actually hands KaTeX, not what we wrote.
    items += [{"id": f"inline L{line_of(outside, s)}", "tex": GH_STRIP_RE.sub(r"\1", t),
               "display": False} for s, t in inlines]
    for r in katex(items):
        if not r["ok"]:
            fails.append(f"KaTeX error [{r['id']}]: {r['error']}")

    for s, t in fences + inlines:
        for mac in BAD_MACROS:
            if re.search(r"\\" + mac + r"(?![A-Za-z])", t):
                fails.append(f"blocked macro \\{mac} in math at L{line_of(text, s)}")

    html = github_render(text)
    n_disp = html.count('js-display-math')
    n_pre = html.count('<pre lang="math"')
    if n_disp != len(fences):
        fails.append(f"display-math placement: {n_disp} rendered vs {len(fences)} written")
    if n_pre:
        fails.append(f"{n_pre} display equation(s) degraded to a code block")

    for m in re.finditer(r"(.?)\(plots/[^)]+\.png\)", text):
        if m.group(1) != "]":
            fails.append(f"un-embedded plot path at L{line_of(text, m.start())}")
    n_img = len(re.findall(r"!\[[^\]]+\]\(plots/[^)]+\.png\)", text))

    for ln in unmotivated_tables(outside):
        fails.append(f"table at L{ln} has no prose sentence above it stating the claim "
                     f"(rule 9a)")

    tics = contrast_tics(outside)
    if len(tics) > CONTRAST_BUDGET:
        fails.append(f"{len(tics)} mechanical contrast constructions, budget is "
                     f"{CONTRAST_BUDGET} (rule 9d): say what the method does, not what "
                     "it isn't\n" + "\n".join(f"    L{ln}: {s[:100]}" for ln, s in tics))

    print(f"{path}: {len(fences)} display eqs, {len(inlines)} inline eqs, "
          f"{n_img} embedded figures, {len(fails)} problem(s)")
    for f in fails:
        print("  FAIL:", f)
    return not fails


if __name__ == "__main__":
    ok = all(check(p) for p in sys.argv[1:])
    print("ALL CHECKS PASS" if ok else "PROBLEMS FOUND")
    sys.exit(0 if ok else 1)
