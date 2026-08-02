I think the current REPORT reaches the wrong conclusion: the character-level runs actually show the Figure 9 pattern.

* Fresh char: LC (1940 \rightarrow 491) at step 15, rises to (769) at step 56, then falls to (8.1) by 30k.
* Pilot char: LC (1940 \rightarrow 484) at step 19, rises to (1043) at step 33, then falls to (68) by 3.5k.

The main bug is in `fig9_verdict.py`: `np.argmin(lc)` finds the global minimum, which usually occurs at the end of the second descent. Please replace this with detection of:

1. the first significant local minimum;
2. the subsequent local maximum;
3. a sustained second descent after that maximum.

Also implement the checks described in the preregistration but currently missing: whether the second descent begins before the clean-accuracy peak, and whether adversarial robustness rises during/after the second descent. Checking only `final_adv_acc >= 0.05` is insufficient.

Please first rerun the corrected verdict on the existing JSON—do not extend training yet—and update
