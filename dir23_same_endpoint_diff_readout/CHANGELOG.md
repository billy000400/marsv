# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

---

## 2026-08-12 — first and final content for REPORT.md and RESULTS.md

Both files went from template stubs to complete deliverables; no earlier numbers were superseded.

- **REPORT.md** — new. Research question, methods (prompt, five readout suffixes, slerp
  interpolation, endpoint JSD, normalized logit distance d(t), t10/t50/t90, width w, Δt50), seven
  embedded figures (immediate prediction; the five individual d(t) curves; transition comparison),
  Table 1, and the verdict. Headline: Δt50 = 0.011 across Capital (t50 0.454), Continent (0.444),
  Currency (0.443) and Language (0.450) → **aligned transitions**. Type control discussed separately.
- **RESULTS.md** — new. S1 endpoint reproduction against the PLAN's preliminary table (all values
  match: immediate JSD 0.00761 bits; readout JSDs 0.991 / 0.885 / 0.915 / 0.968 / 0.111 bits), full
  transition table with crossing counts and monotonicity, top-1 flip positions (t = 0.44–0.47), the
  immediate-position d(t) caveat, the overlay figure, and the data/code index.
- **plots/** — new: distance_{capital,continent,currency,language,type}.png, distance_overlay.png,
  immediate_prediction.png, transition_comparison.png.
- **results/** — new: s1_endpoints.json, interp.csv, interp.npz, transitions.json.
