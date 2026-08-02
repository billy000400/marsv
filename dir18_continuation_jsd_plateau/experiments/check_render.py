"""Check that a Markdown deliverable actually renders on GitHub.

Four checks, all of which have caught a real bug in this direction:

1. KaTeX compile of every display (```math fence) equation.
2. KaTeX compile of every inline $...$ expression *after* simulating GitHub's
   backslash-before-punctuation stripping (CLAUDE.md rule 8b).
3. Macros GitHub's markdown math renderer rejects outright ("The following macros
   are not allowed: ..."), e.g. \\operatorname -- use \\mathrm / \\text instead.
4. Display-math placement via the GitHub markdown API (rule 8a) + figure embeds (rule 12).

Usage: python experiments/check_render.py REPORT.md RESULTS.md
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

    print(f"{path}: {len(fences)} display eqs, {len(inlines)} inline eqs, "
          f"{n_img} embedded figures, {len(fails)} problem(s)")
    for f in fails:
        print("  FAIL:", f)
    return not fails


if __name__ == "__main__":
    ok = all(check(p) for p in sys.argv[1:])
    print("ALL CHECKS PASS" if ok else "PROBLEMS FOUND")
    sys.exit(0 if ok else 1)
