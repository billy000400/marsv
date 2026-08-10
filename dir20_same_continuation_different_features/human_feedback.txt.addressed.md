Mathew used GPT-2 Large (36 layers); this report uses GPT-2 Medium (24 layers). Since plateau sharpness depends strongly on downstream depth, GPT-2 Medium is not a reproduction. The different big/in curve may simply be model-dependent. Mathew’s post
More importantly, big/in is Mathew’s positive plateau example, not a negative control. His non-plateau comparison is big/large, as confirmed by the original configuration.
The hypothesis was also changed. Your advisor’s hypothesis is:
holding output JSD low, different circuits/features may occupy different plateaus.
It does not predict that lower JSD should produce sharper plateaus. Therefore correlating JSD with transition width—and arguing that big/in falsifies the hypothesis—tests the wrong claim. The missing independent variable is an actual measurement of circuit/feature difference.
“Depth, not prompt content, produces the plateau” is too strong. Depth explains how a boundary becomes sharper, but whether sharpening occurs depends on the interpolation path. Mathew’s big/large remains smooth despite having even more downstream layers.
The “plateaus everywhere” claim is numerically overstated: under the predefined w 
10−90
​	
 <0.5 criterion, only 5/10 cases qualify; under the report’s w 
TV
​	
 <0.25 criterion, 6/10 qualify—not 9/10. Current report