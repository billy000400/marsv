#!/usr/bin/env python3
"""check_md.py — verify Markdown will render on GitHub (format gate only).

Usage:  python check_md.py FILE [FILE ...]
        python check_md.py REPORT*.md RESULTS.md
Exit 0 = all clean (warnings allowed), 1 = at least one ERROR, 2 = usage.

Passing this script is not completion; run.sh also requires a fresh content review.

Checks (the high-frequency GitHub-render failures):
  E1  display math $$ unbalanced (odd count)
  E2  LaTeX delimiters \\[ \\] \\( \\) present  -> GitHub won't render; use $$...$$ or $...$
  E3  image embed ![](path) points to a file that doesn't exist on disk
  W1  a plots/*.png is named in prose but never embedded as ![](...)  -> it won't display
"""
import sys, os, re

def strip_code(s: str) -> str:
    s = re.sub(r'(?ms)^(```|~~~)[^\n]*\n.*?^\1\s*$', '', s, flags=re.S)  # backtick or tilde fenced blocks
    s = re.sub(r'`[^`]*`', '', s)                 # inline code
    return s

def check(path):
    errs, warns = [], []
    try:
        src = open(path, encoding='utf-8').read()
    except FileNotFoundError:
        return [f'file not found: {path}'], []
    base = os.path.dirname(os.path.abspath(path))
    body = strip_code(src)

    # E1 balanced $$
    n = body.count('$$')
    if n % 2:
        errs.append(f'unbalanced $$ (found {n}; must be even)')

    # E2 forbidden LaTeX delimiters (single backslash, not \\[ )
    if re.search(r'(?<!\\)\\[\[\]()]', body):
        errs.append(r'contains \[ \] \( or \) — GitHub does not render these; use $$...$$ or $...$')

    # E3 image embeds resolve on disk
    embedded = re.findall(r'!\[[^\]]*\]\(([^)\s]+)', body)
    for p in embedded:
        if p.startswith(('http://', 'https://')):
            continue
        if not os.path.exists(os.path.join(base, p)):
            errs.append(f'image path not found: {p}')

    # W1 figures named in prose but not embedded
    linked = set(embedded) | set(re.findall(r'\]\(([^)\s]+)', src))
    for p in set(re.findall(r'(plots/[\w./-]+\.png)', body)):
        if p not in linked:
            warns.append(f"figure '{p}' named in text but not embedded as ![](...) — it won't display")

    return errs, warns

def main():
    files = sys.argv[1:]
    if not files:
        print('usage: check_md.py FILE [FILE ...]'); sys.exit(2)
    bad = False
    for f in files:
        e, w = check(f)
        for x in w: print(f'WARN  {f}: {x}')
        for x in e: print(f'ERROR {f}: {x}')
        if e: bad = True
        if not e and not w: print(f'OK    {f}')
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()