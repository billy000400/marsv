# CLAUDE.md — operator rules for every agent in this project

> Read and FOLLOW these rules every iteration, in addition to BUDGET.md and the direction's PLAN.md.
> Hard rules: when a rule conflicts with convenience or speed, the rule wins. Shared across all
> directions; lives at the project root.
>
> **Autonomous-loop mode:** you run headless with no human to answer mid-iteration. Wherever a rule
> below says "ask" or "stop and clarify", instead: pick the most standard option, record the
> assumption AND the alternatives you rejected in JOURNAL.md, and continue. Never block the loop
> waiting for input.
>
> **Version control is automatic — do NOT run git yourself.** After every iteration the wrapper
> (`run.sh`) commits this direction's work and pushes it to GitHub, serialized across the concurrent
> agents. Just persist your work to disk; the wrapper handles the rest. (If a git push ever fails on
> SSH/auth, the wrapper self-heals by running `/mars-vol/setup_github_ssh.sh` and retrying.)

## File roles — know which are CURATED vs APPEND-ONLY
| File         | Role                           | How to write it                                          |
|--------------|--------------------------------|---------------------------------------------------------|
| RESULTS.md   | Final, presentable deliverable | CURATE: read, then overwrite to current-best. No history.|
| REPORT.md    | Final, presentable deliverable | CURATE: read, then overwrite to current-best. No history.|
| CHANGELOG.md | History of deliverable changes | APPEND-ONLY. Never rewrite earlier entries.             |
| JOURNAL.md   | Working log                    | APPEND-ONLY. Never rewrite earlier entries.             |
| PLAN.md      | Live plan                      | Edit in place (status / next-step / checkboxes).        |

---

# Part A — Engineering discipline (applies to the code you write under experiments/)

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

(Loop mode: you can't ask — log the assumption + rejected alternatives in JOURNAL.md and proceed.)

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.
(This governs CODE. The research DELIVERABLES — RESULTS.md/REPORT.md — are deliberately CURATED, not
surgical: see Part B. Code = minimal diffs; reports = rewritten clean to current-best.)

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

# Part B — Files, results & reports (research deliverable hygiene)

## 5. Read before write.

Never overwrite or edit a file without first reading its current contents. Prefer a targeted edit
over a full rewrite; preserve everything you are not deliberately changing. Never blank-truncate a
file you have not read. (For RESULTS.md/REPORT.md, "read then rewrite clean" is expected per rule 6 —
but you still READ first.)

## 6. RESULTS.md and REPORT.md are finalized deliverables — curate, don't log.

Always overwrite them to reflect the CURRENT BEST state, as if for a reader seeing them for the first
time. NEVER keep version history, "changed after review" notes, "v1/v2", or multiple runs of the SAME
experiment in them. When a stronger or more statistically significant result for the same experiment
appears, REPLACE the weaker one — do not show both. One experiment -> one current result.

## 7. All history lives in CHANGELOG.md (append-only).

Each iteration, append a dated entry recording what changed in RESULTS.md/REPORT.md and why — and if a
result was superseded, the old -> new numbers. This is the ONLY place result/version history belongs.

## 8. REPORT.md must be a self-contained, presentable report.

Structure: `Summary -> Methods -> Results -> Conclusion`. The **Methods** section MUST state:
- **Data & Model:** dataset, model (e.g. GPT-2 small, 124M), and exactly which layer(s)/hook point and sample sizes.
- **Metrics:** every metric defined with a RENDERED equation (see 8a for the correct fence), stating exactly what is scored.
- **Baselines:** every baseline named and defined (equation where applicable).
Results show current-best numbers only, with figures referenced from plots/.

### 8a. Display math must live at COLUMN 0 (top level), never nested in a list item.

**The rule, verified against GitHub's own renderer (`POST https://api.github.com/markdown`):**
keep every display equation — whether `$$…$$` or a ` ```math ` fence — as its own top-level block at
**column 0, with a blank line before and after**. Do NOT put display math inside a `-`/`*`/`1.` list
item. Inline `$…$` in a sentence is fine for its *placement* — but its *contents* have a separate
trap; see **8b**.

**Why (the exact failure modes, both confirmed by reproduction):**
1. A `$$…$$` block **glued** to the end of the preceding prose line (no blank line before `$$`) is
   parsed as inline text and GitHub dumps the raw LaTeX — at column 0 too, not just in lists.
2. **The subtle one that bit us:** an **indented ` ```math ` fence inside a list item renders as a
   plain code block (gray box + copy icon), not math, whenever that same list item's text contains
   any inline `$…$`** (e.g. `- **metric** with score $h$ …:` followed by a fence). Even one inline
   `$…$` pair triggers it. Our Methods bullets are full of `$h$`, `$k=4$`, `$v_i$`, so **every** fence
   silently degraded to a code block. An indented fence in a list item with NO inline `$…$` happens to
   render, but do not rely on that — the safe rule is simply: **no display math inside list items.**

**So don't nest.** Write Methods/Baselines method entries as **bold run-in paragraphs**, not bullets,
so the equation sits at column 0. Pattern (outer `~~~` only so the inner fence shows literally):

~~~
**metric** — one-line description; inline `$h$`, `$k$` here are fine, ending in a colon:

```math
s(x) = \lVert \nabla_h \log p \rVert_F
```

Follow-up prose as its own paragraph.
~~~

Prefer the ` ```math ` fence over `$$…$$` for display blocks (no glued-delimiter trap), but at column 0
either works. **Verify before committing:** pipe the file through the GitHub markdown API and confirm
every display equation becomes a `math-renderer class="js-display-math"` element and NONE become
`<pre lang="math">` code blocks:

~~~
python3 -c "import json;print(json.dumps({'mode':'gfm','text':open('REPORT.md').read()}))" \
  | curl -s -X POST -H "Accept: application/vnd.github+json" -d @- https://api.github.com/markdown \
  | grep -c 'js-display-math'   # must equal your display-equation count; grep -c '<pre lang="math"' must be 0
~~~

This check confirms *placement* (delimiters recognized) but does **not** compile the LaTeX — so it will
pass even when the equation itself is broken by **8b**. Do both checks.

### 8b. Inside inline `$…$`, GitHub strips the backslash before ASCII punctuation — so `\_ \{ \} \| \, \; \!` etc. break.

Verified against `POST api.github.com/markdown`: GitHub applies Markdown backslash-escaping to the
*inside* of an inline `$…$`. Any backslash immediately followed by ASCII punctuation
(`_ { } | , ; ! : % # & ~ ^` …) has its backslash removed **before** the LaTeX reaches KaTeX. Fenced
` ```math ` blocks are literal and do NOT suffer this — everything is preserved.

Consequences (each reproduced): inline `$\texttt{plateau\_auc\_low}$` → KaTeX sees `\texttt{plateau_auc_low}`
→ red error `'_' allowed only in math mode`; inline `$\min\big\{x\big\}$` → `\min\big{x\big}` → broken
`\big{`; inline `$\|h\|$` → `|h|` (single bars, wrong norm); inline `$a\,b$` → `a,b` (stray comma, lost
thin-space). None of these show up in the js-display-math check above — they fail silently or as a KaTeX
box in the browser.

**Rules for inline math:**
- If the expression needs any `\`-escaped punctuation, **put it in a ` ```math ` fence** (at column 0),
  not inline. Fences preserve everything.
- If it must stay inline (table cell, mid-sentence), use the backslash-**letter** macro that survives
  Markdown: `\{`→`\lbrace`, `\}`→`\rbrace`, `\|`→`\Vert` (and `\lVert`/`\rVert`), `\,`→`\thinspace`,
  `\;`→`\thickspace`, `\!`→`\!`… has no letter form, so drop it or fence. Bare subscripts in math mode
  (`x_c`, `\lambda_i`) are fine — the trap is only a *backslash* before punctuation.
- **Grep before committing** for the hazard in inline math (outside fences):
  `grep -nP '\$[^$\n]*\\[,;!:{}|_%#&][^$\n]*\$' REPORT.md` — every hit is a latent break.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying assumptions are logged before implementation rather than discovered after mistakes, and RESULTS.md/REPORT.md always read as a clean current-best paper with all history in CHANGELOG.md.
