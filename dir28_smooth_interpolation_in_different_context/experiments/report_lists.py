"""Write the complete W_final > 0.3 lists and the cross-context set lists as collapsible Markdown fragments.

Reads results/<ctx>_W_gt_0.3.csv and results/cross_context.csv (written by analyze.py);
writes results/fragments/*.md, which are pasted verbatim into REPORT.md's appendix.
Tokens are shown as Python repr() inside a text fence so control bytes and backticks cannot break Markdown.
"""
import csv
import os

from analyze import CTX, PROMPTS, RES

OUT = os.path.join(RES, "fragments")
os.makedirs(OUT, exist_ok=True)


def fence(lines):
    return "```text\n" + "\n".join(lines) + "\n```\n"


def main():
    for k, c in enumerate(CTX, 1):
        rows = list(csv.DictReader(open(os.path.join(RES, f"{c}_W_gt_0.3.csv"))))
        lines = [f"{'rank':>4}  {'W':>5}  {'D':>5}  token"] + [
            f"{r['rank']:>4}  {float(r['W_final']):.3f}  {float(r['D']):5.2f}  {r['token']}"
            + ("  [flagged]" if r["flag_nonmonotonic"] == "1" else "") for r in rows]
        with open(os.path.join(OUT, f"list_{c}.md"), "w") as f:
            f.write(f"<details>\n<summary>Section {k} (\"{PROMPTS[c]}\"): all {len(rows)} tokens with "
                    f"W &gt; 0.3, largest first</summary>\n\n{fence(lines)}\n</details>\n")

    rows = list(csv.DictReader(open(os.path.join(RES, "cross_context.csv"))))
    groups = [("all four contexts", [r for r in rows if r["n_ctx"] == "4"])]
    groups += [(f"exactly {n} contexts", [r for r in rows if r["n_ctx"] == str(n)]) for n in (3, 2)]
    for k, c in enumerate(CTX, 1):
        groups.append((f"only Section {k} (\"{PROMPTS[c]}\")",
                       [r for r in rows if r["n_ctx"] == "1" and float(r[f"W_{c}"]) > 0.3]))
    with open(os.path.join(OUT, "cross.md"), "w") as f:
        for name, g in groups:
            g = sorted(g, key=lambda r: -float(r["max_W"]))
            lines = ["  max_W  " + "  ".join(f"W_{c}" for c in CTX) + "  token"] + [
                f"  {float(r['max_W']):.3f}  " + "  ".join(f"{float(r[f'W_{c}']):.3f}" for c in CTX)
                + f"  {r['token']}" for r in g]
            f.write(f"<details>\n<summary>Tokens with W &gt; 0.3 in {name}: {len(g)}</summary>\n\n"
                    f"{fence(lines)}\n</details>\n\n")


if __name__ == "__main__":
    main()
