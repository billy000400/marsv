 reviewed the code and report. The core result is reproducible—(\rho(JSD_{\text{corpus}}, w)=-0.419), (p=1.81\times10^{-4})—but the current version needs a major revision before the conclusions are reliable. Please make the following changes:

1. Fix `width()` validity checking. It currently finds only the first upward 0.1/0.9 crossings and can accept non-monotonic or multi-transition curves. Implement the validity criteria defined in the plan, report the invalid-curve rate, and save/commit all raw (d(t)) curves. Figure 2’s pointwise median curves are not sufficient for auditing.

2. Rebuild the primary pair bank using top-256 tokens exactly as prespecified, even if this gives fewer than 75 pairs. The top-512 analysis may remain only as a clearly labeled post-hoc secondary analysis; do not call it a prespecified fallback.

3. Correct the interpretation. The current evidence shows that corpus JSD predicts learned output separation and overall transition width:

   * (\rho(JSD_{\text{corpus}},JS
