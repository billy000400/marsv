# Human feedback: simplify the report and separate correspondence from formation

Please revise `REPORT.md` and `RESULTS.md` only. `PLAN.md` is operator-owned; do not edit it. Preserve all existing code, logs, plots, and result files even when they are removed from the current report.

The current directory should answer only:

> Do token pairs with more different immediate-next-token distributions tend to have narrower transitions in the trained model's output-distance score `d(t)`?

## Keep

- Lead with the final-checkpoint **1,000-pair result**, because it shows the overall relationship most clearly. State that tokens are reused and the uncertainty calculation accounts for this.
- Follow with the controlled **60-pair result**, where no token is reused and frequency/surprisal are matched.
- Keep corpus-JSD reliability, model-output-JSD validation, curve validity, and the limitation that `w` and edge drift are almost the same measurement.
- Keep step 0 only as a brief baseline, and 410M only as a cross-scale check.
- Keep the final-checkpoint adjustment result: the overall association is strong, but is no longer significant after accounting for model-output divergence and all measured pair properties.
- Keep `big`/`large` and `big`/`in` as illustrations. State that `d(t)` is uninformative when endpoint outputs are already almost identical.
- Put pair construction, split checks, alternative banks, full statistical details, and the 60-pair table in Methods or an Appendix.

## Remove or move out

- All intermediate checkpoints, the formation figure/table, onset claims, the 64k-to-final reversal, and detailed learned `Î”w`. These belong in the formation directory.
- The block scan from the main story. It may remain as a short Appendix sanity check, but it does not answer the correspondence question.
- Audit history and feedback-round narration; history belongs in `CHANGELOG.md`.
- Any claim that a plateau represents one continuation distribution or that continuation distributions jump at the boundary. Only the two endpoints were measured.

## Jargon replacements

| Avoid in the main text | Use instead |
|---|---|
| held-out JSD | next-token JSD measured on a separate corpus sample |
| selection/holdout split | pair-selection sample / measurement sample |
| stratified pair bank | pairs chosen to cover the full JSD range |
| endpoint-disjoint | no token is reused across pairs |
| carrier context | fixed sentence frame |
| quintile or stratum | one of five JSD groups |
| assay | interpolation experiment |
| relative-logit coordinate | output-distance score `d(t)` |
| 10%-90% transition width | fraction of the path needed for `d(t)` to move from 0.1 to 0.9 |
| dyadic endpoint bootstrap | uncertainty calculation that accounts for tokens reused across pairs |
| mediator/covariate adjustment | after accounting for model-output difference and the measured pair properties |

Define JSD and `w` once in plain English. Do not say that â€œthe full output stays putâ€: a flat `d(t)` only means that this relative endpoint-distance score changes slowly.

Use this order: question and conclusion; 1,000-pair result; controlled 60-pair result; essential checks; examples; limitations; Methods/Appendix.

The final claim should be no stronger than:

> Across a large 1,000-pair analysis and a controlled 60-pair analysis, tokens with more different immediate-next-token distributions tend to have narrower transitions in the trained model's output-distance score. This is an observational endpoint-level relationship; it does not show that each plateau corresponds to one continuation distribution or that corpus JSD causes the transition.