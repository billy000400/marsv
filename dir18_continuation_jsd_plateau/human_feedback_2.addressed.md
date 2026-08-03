Conclusion: minor revision. The core scientific result has been reproduced, but the report and auditable artifacts haven't fully passed yet.
I reran the top-256 primary bank on the latest commit c03d510:
ρ(JSDB, w) = −0.525, p = 1.68×10⁻⁵, n = 60
step 0: −0.056, p = 0.672
Top-256, endpoint-disjoint, and the new validity code are all correctly implemented.
Remaining issues, by severity:
Raw curves still not committed. The report claims three times that curves_*.npy/.csv.gz have been committed, but none exist in the repo; .gitignore still ignores *.npy and *.gz. As a result, the QC numbers can't be independently recomputed. Either force-add/unignore the curves, or remove the incorrect "committed" statements.
The requested mediator and learned-sharpening analyses are still missing. I reran and got: ρ(JSDB, w_trained − w_step0) = −0.517, p = 2.31×10⁻⁵, 95% CI [−0.695, −0.297] After controlling for output JSD: −0.277, p = 0.032; after further controlling for five covariates: −0.204, p = 0.119. The report should clearly state that the headline number is the total association, and the fully-adjusted independent relationship is not significant.
The training-dynamics text is still wrong. Figure 8 still says "sharpens throughout training," but w rebounds from 0.512 at 64k to 0.541 at final; 38/60 pairs became blunter, paired Wilcoxon p = 0.0052. This should read "sharpens through 64k, followed by a modest late reversal."
The "complete-word filter" still isn't implemented. common.py still only checks Ġ, lowercase letters, and length; the primary bank still contains un. After removing that pair, results remain robust: ρ = −0.502, p = 5.15×10⁻⁵ So this can simply be included as a sensitivity check, with the description changed to "word-start tokens."
These revisions don't change the core positive result. Once the raw curves are committed, the two missing statistics are added, and the wording is corrected, I'll approve.