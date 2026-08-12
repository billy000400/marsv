# CLAUDE.md — operator rules for every agent in this project

> Read and FOLLOW these rules every iteration, in addition to BUDGET.md and the direction's PLAN.md.
> Hard rules: when a rule conflicts with convenience or speed, the rule wins. Shared across all
> directions; lives at the project root.
>
> **Autonomous-loop mode:** you run headless with no human to answer mid-iteration. Wherever a rule
> below says "ask" or "stop and clarify", use a standard documented assumption only for ordinary
> implementation details. A materially ambiguous scientific request or output path is different:
> record it as `blocked` in the active task manifest and stop that task. Never substitute a nearby
> research question, rewrite literal data such as `&` as the word “and,” or choose `REPORT.md`
> because the requested destination is unclear.
>
> **Version control is automatic — do NOT run git yourself.** After every iteration the wrapper
> (`run.sh`) commits this direction's work and pushes it to GitHub, serialized across the concurrent
> agents. Just persist your work to disk; the wrapper handles the rest. (If a git push ever fails on
> SSH/auth, the wrapper self-heals by running `/mars-vol/setup_github_ssh.sh` and retrying.)

## File roles — know which are CURATED vs APPEND-ONLY
| File         | Role                           | How to write it                                          |
|--------------|--------------------------------|---------------------------------------------------------|
| RESULTS.md   | Final, presentable deliverable | CURATE: read, then overwrite to current-best. No history.|
| REPORT*.md   | Final, presentable deliverables| CURATE only the requested file; no history.              |
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
(This governs CODE. The research DELIVERABLES — RESULTS.md/REPORT*.md — are deliberately CURATED, not
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
file you have not read. (For RESULTS.md/REPORT*.md, "read then rewrite clean" is expected per rule 6 —
but you still READ first.)

## 6. RESULTS.md and every REPORT*.md are finalized deliverables — curate, don't log.

Always overwrite them to reflect the CURRENT BEST state, as if for a reader seeing them for the first
time. Edit only the report path requested by the task manifest. NEVER keep version history, "changed
after review" notes, "v1/v2", or multiple runs of the SAME
experiment in them. When a stronger or more statistically significant result for the same experiment
appears, REPLACE the weaker one — do not show both. One experiment -> one current result.

## 7. All history lives in CHANGELOG.md (append-only).

Each iteration, append a dated entry recording what changed in RESULTS.md/REPORT*.md and why — and if a
result was superseded, the old -> new numbers. This is the ONLY place result/version history belongs.

## 8. Every REPORT*.md must be a self-contained, presentable report.

`WRITING.md` is the canonical short writing policy and is appended by `run.sh` through Claude's
system-prompt-file mechanism. It applies equally to `REPORT.md`, `REPORT_followup.md`, and supplements.

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

### 8c. GitHub REJECTS some macros outright. `\operatorname` is the one that keeps biting us.

GitHub runs KaTeX with a macro **denylist**, and one denied macro replaces the whole equation with the
red error **"The following macros are not allowed: operatorname"** — the formula is simply gone. The
LaTeX is valid and both checks above pass, which is why an operator has now reported this twice
(dir13 feedback #1 and #4, the second a regression introduced by a later rewrite).

- Never `\operatorname{…}` (nor `\operatorname*`, `\DeclareMathOperator`). Write `\mathrm{softmax}` or
  `\text{softmax}` instead. Built-in operators (`\max`, `\min`, `\arg\max`, `\Pr`, `\sin`, `\arccos`,
  `\exp`, `\log`) are fine and need no wrapper.
- Never definition macros (`\def \gdef \edef \xdef \let \newcommand \renewcommand \providecommand`)
  or HTML/link macros (`\href \url \includegraphics \htmlClass \htmlId \htmlStyle \htmlData`).

### 8d. Run ONE script that checks 8a–8c, rule 12 and rules 9a/9d. Eyeballing has failed every time.

`dir13_plateau_on_grok_gpt/experiments/check_render.py` (with `katex_compile.js`) does all four checks
and exits non-zero on any problem: it compiles every ` ```math ` fence with KaTeX, compiles every
inline `$…$` **after applying GitHub's backslash-stripping** (so 8b breaks surface as real KaTeX
errors), flags every denylisted macro, and confirms via the GitHub API that each display equation
became `js-display-math` and none became `<pre lang="math">` — plus that no `(plots/x.png)` path is
missing its `![…]` embed, that every table has a real prose paragraph above it (rule 9a), and that
self-describing contrast constructions stay within budget (rule 9d). One-time setup:
`npm install --prefix /tmp/katexcheck katex`. Copy it into your direction and run it before you
finish an iteration:

~~~
python3 experiments/check_render.py REPORT*.md RESULTS.md  # exit 0 = renders on GitHub
~~~

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying assumptions are logged before implementation rather than discovered after mistakes, and RESULTS.md/REPORT*.md always read as clean current-best deliverables with all history in CHANGELOG.md.

## 9. Write RESULTS.md and every REPORT*.md for a newcomer to technical AI safety research.

Key points it instructs agents to follow:
- **Reader model:** a strong ML engineer / first-year grad student who has *not* read the subfield's papers.
- **Lead with the "why"** — open with the safety question and stakes before method detail.
- **Define jargon on first use** and spell out acronyms (OOD, SAE, AUROC, probe, logit lens, …).
- **Explain every metric in words next to its equation** (what it measures, how to read it, higher-is-better, etc.) — additive to rule 8's equation requirement.
- **Motivate every metric BEFORE defining it** — one or two sentences on what question this metric answers and why the obvious alternative doesn't work (e.g. "softmax confidence saturates under MSE training, so we use max raw output"). Then say which Result/figure consumes it. Methods must read as a narrative ("to answer X we measure Y"), never as a bare list of definitions. A metric no Result uses gets cut.
- **Interpret the numbers** — say whether a result is strong/weak/surprising and what it implies.
- **Plain words, short sentences, active voice.**
- **Keep rigor intact** — accessible means well-explained, not vague; don't drop caveats/CIs/sample sizes to sound friendlier.
- A one-line **test**: could a capable ML engineer new to the subfield read the affected REPORT*.md and correctly explain what you did, why it matters, and what you found, without another source?

### 9a. State the strengths explicitly. Never make the reader infer the contribution from a table.

A table of numbers is evidence, not an argument. Wherever you present a favorable result, say in prose:
**where** the method performs best (which regime, model, layer, data slice), **which measured result**
supports that statement, **what practical problem** it addresses, and **why that capability matters**
relative to existing methods. Explain why an advantage arises only when causal evidence supports that
explanation; a checkpoint correlation is not a mechanism. "Ours: 0.87 / baseline: 0.71" with no
surrounding sentence is an unfinished result.

**Test:** delete every table and figure from the affected REPORT*.md — can a reader still name your contribution and why
it is useful? If not, the prose is doing too little. `check_render.py` (rule 8d) enforces the mechanical
half of this: a table sitting directly under a heading, with no prose paragraph above it, fails.

### 9b. Let the story follow the evidence, not the other way round.

This applies only to self-directed framing after all requested analyses are complete. It never permits
changing a human's research question, output routing, or checklist to fit a more favorable story.

When the results do not support the narrative you started with, **rebuild the narrative around the strongest
evidence you actually have** — redefine the problem, reorder the contributions, pick a new primary result, and
restructure the Summary, motivation, and Results sections to match. Do not keep a Summary that promises what
the numbers no longer deliver, and do not bury the strongest finding in a subsection because an earlier plan
called it secondary. In loop mode, record the re-framing (old story → new story, and what evidence forced it)
in JOURNAL.md and CHANGELOG.md; each REPORT*.md itself shows only the current story (rule 6).

### 9c. Use this default order when the results are unfavorable.

Work down the list and stop at the first step that honestly fixes the mismatch:

1. **Cut** content unrelated to the core claim.
2. **Narrow** the claim to what the evidence supports.
3. **Adopt a more informative evaluation dimension** — one that actually distinguishes the methods.
4. **Explain differences in objectives or trade-offs** — only when the evidence supports that explanation.
5. **Reorganize the experiments** so the genuine strengths come first.
6. **Redefine the paper's overall story** (rule 9b).
7. **Discuss the limitation** — only when it cannot be avoided and it materially affects the central conclusion.

This is a presentation order, not permission to hide anything: steps 1–6 reframe what the paper *claims*,
they never delete a result that contradicts a claim you are still making. A caveat, CI, or sample size that
qualifies a surviving claim stays (rule 9's "keep rigor intact").

### 9d. Avoid mechanical contrast constructions.

Do not repeatedly write "We do X rather than Y", "A is X, not Y", "unlike prior work, we …". One or two such
sentences in a whole report is fine; a paragraph built out of them reads as defensive and says less than it
appears to. State directly what the method does, why it is designed that way, and what the design achieves —
the contrast is then obvious without being announced.

`check_render.py` (rule 8d) counts these and fails above a budget of 2 per file. It flags only sentences
that describe *our own* work by contrast — "we measure X rather than Y", "our detector scores X instead of
Y", "unlike prior work, …". An ordinary comparison between two things being studied ("the variance is
explained by per-character terms rather than pair chemistry") is not the tic and is not flagged.

---

# Part C — Operator feedback & plot embedding (applies to EVERY direction, EVERY iteration)

## 10. Address operator feedback before anything else.

Humans drop feedback into this direction as files beginning `human_feedback` (any extension), or
containing `REVIEW`. An unaddressed feedback file is any such file not ending in `.addressed.md`.
`workflow.py prepare` creates `.tasks/<source>.manifest.json` before any research begins.

The manifest is the compact task authority. It contains the exact source filename, required output
files, existing files that may change, files that must remain unchanged, unresolved ambiguities, and
a verbatim checklist. Never edit a checklist request. In particular, literal characters such as `&`
must remain literal. Copy user paths exactly and never redirect a follow-up into `REPORT.md`.

The first agent pass is triage only: fill routing fields and verify the checklist before research.
If a scientific instruction or destination is materially unclear, set the manifest to `blocked`,
record the ambiguity, and stop. Otherwise set it to `ready`. Later passes fill, for every item:
what was done; the exact figure, table, experiment, report section, or file answering it; and whether
ambiguity remains. Only a complete task may be set to `review_pending`.

The worker must never rename feedback. After render/format checks, `run.sh` starts a fresh content
reviewer with only the original feedback, manifest checklist, and every declared or modified report
and required output. The reviewer rejects omitted or reinterpreted asks, a changed research question,
displacing requested work with unrequested analysis, undefined jargon, causal language unsupported by
correlation, claims stronger than evidence, or prose inaccessible to a new researcher. Passing render
checks is necessary but not sufficient. Only the wrapper may rename approved feedback to
`.addressed.md`; infeasible or ambiguous items remain unaddressed.

## 11. Never STOP while unaddressed feedback remains.

The wrapper's loop halts the moment a `STOP` file exists. So **do NOT write `STOP` if any
`human_feedback*` / `*REVIEW*` file without `.addressed.md` is present** — a STOP'd direction stops
looping and will silently ignore feedback dropped afterward. Only write `STOP` once (a) the plan is
complete AND (b) zero unaddressed feedback files remain. If you re-enter and find new unaddressed
feedback next to a stale `STOP`, delete `STOP`, address the feedback, and only re-write `STOP` when
clean again.

## 12. Embed every plot as a RENDERED image in every affected REPORT*.md and RESULTS.md.

A bare path like `(plots/foo.png)` in prose does NOT render — it is just text, and the figure never
appears. **Every quantitative result must be embedded as an actual Markdown image** so it renders on
GitHub:

```
![Short descriptive caption](plots/foo.png)
```

- This applies to every affected REPORT*.md, not only RESULTS.md, and on **normal iterations**, not only at
  finalization: whenever you curate the deliverables, (re)embed the current-best figures as `![](…)`
  images. Do not defer report figures to the last 20 minutes.
- Grep before committing to catch un-rendered path references:
  `grep -nE '\(plots/[^)]+\.png\)' REPORT*.md RESULTS.md` — every hit that is not preceded by `!` and
  a caption in `![...]` is a figure that will NOT render. Convert it to an `![caption](plots/….png)`
  embed (a parenthetical pointer alongside an already-embedded figure is fine).
- **Motivate every figure BEFORE it appears, and cut the ones nothing needs** (the figure analogue of
  rule 9's metric rule). Each embedded plot must be preceded by prose naming the question it answers or
  the claim it evidences — "to show X, we plot Y" — not merely describing what it depicts. Two figures
  in a row with no prose between them means the second is unmotivated: either give it its own
  sentence or drop it. **A figure no claim in Results depends on gets CUT, not embedded** — leave the
  PNG in plots/ if you like, but it does not enter the deliverables. Coverage ("a PNG per quantitative
  result") governs what you *save*; this rule governs what you *show*.
- **ALT TEXT IS NOT A CAPTION — it does not render.** GitHub shows the `![...]` text only when the
  image fails to load. A caption written inside the brackets is INVISIBLE to every reader. (This has
  already cost us a whole report: dir13 put all 16 captions in alt text and rendered 16 unlabeled
  images.) Every figure needs a **visible caption line immediately below the image**, starting with a
  bold figure number:

~~~
![short alt text for screen readers](plots/foo.png)

**Figure 3.** What the figure shows and what to conclude from it. x: interpolation position `t`;
y: relative distance `d(t)`. Solid = between-plateau pairs, dashed = within-plateau controls.
~~~

  Keep the alt text short (one clause); the caption below carries the axes, the series, and the point.
- **Number figures sequentially in reading order, and reference them from the prose.** Figure 3 must
  appear after Figure 2 and before Figure 4 in the file — not renumbered from a deleted draft, not left
  unnumbered, not appended out of order in an appendix (dir13 currently embeds Figures 3–5 *after*
  Figure 11, and two figures with no number at all). Every figure is cited at least once by number in
  the body ("Figure 3 shows…"); a figure the prose never cites is a figure rule 12's motivation clause
  says to cut.
- **Check before finishing** — every embed must be followed by a visible caption line:

~~~
grep -A2 -nE '^!\[' REPORT*.md RESULTS.md | grep -c '\*\*Figure'  # must equal the number of embeds
~~~

- **Every figure must be readable from the report alone.** For each embedded plot, the caption or the
  adjacent prose must state what the x-axis and y-axis are (variable name, units/scale, e.g. log axis),
  and every variable appearing on an axis, in a legend, or as a group label (e.g. "confident",
  "contrast") must be DEFINED in the report's Methods before the figure appears. Before committing,
  check each figure: "could a newcomer name both axes and every legend entry using only this report?"
  If not, the figure is not done.
- Save figures headless (`plt.savefig(...)` + `plt.close()`, never `plt.show()`; `MPLBACKEND=Agg`).

## 13. Plots must be readable with red-green color deficiency. This is a HARD requirement.

**The operator of this project has red-green color deficiency.** A figure whose meaning depends on
telling red from green is unreadable to its primary reader — treat that as a broken figure, exactly
like an unrendered one.

- **Never use red vs green as a contrast.** Not for two series, not for pass/fail, not for
  above/below-baseline. This is the single most common failure and it is already pervasive in this
  project's existing plots.
- **Categorical series — use this palette, in this order** (green-free; validated for deuteranopia,
  protanopia, and tritanopia at worst-pair ΔE 9.6, above the ΔE 8 target, across ALL pairs):

```python
CVD = ["#0072B2",  # blue
       "#D55E00",  # vermillion
       "#CC79A7",  # reddish purple
       "#56B4E9",  # sky blue
       "#E69F00"]  # orange
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CVD)
```

  Take hues in order; do not cycle or invent a 6th. More than 5 series → small multiples, or fold the
  tail into one gray "other".
- **Color alone must NEVER be the only channel carrying identity.** Every series also varies
  `linestyle` (`-`, `--`, `:`, `-.`) or `marker` (`o`, `s`, `^`, `D`); every hatched/filled region also
  varies `hatch` (`//`, `\\`, `..`). Direct-label series at their right-hand end where the plot allows
  it. The test: **would this figure still be readable printed in grayscale?** If not, it is not done.
- **Colormaps.** Sequential: `viridis` or `cividis` (`cividis` is designed for CVD). Diverging:
  `coolwarm` or `RdBu` — blue↔red/orange with a neutral midpoint. **NEVER `jet`, `rainbow`, `hsv`,
  `RdYlGn`, or `Spectral`** — all fail on red-green and none encode magnitude monotonically.
- **Never identify a series by its color in prose or captions.** Write "the between-plateau median
  (dashed)" or "the MSE seeds", not "the red curve" / "the green baseline". A caption that says
  "red and green distributions overlap" conveys nothing to this reader — name the series and the
  non-color channel that distinguishes it.
