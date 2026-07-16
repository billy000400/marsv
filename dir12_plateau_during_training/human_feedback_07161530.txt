Top 5 readability improvements:

1. **Shorten and simplify the Summary.**  
   It is nearly a miniature report, with dense sentences, jargon, and many statistics. Keep the motivation, three principal findings, and verdict; move details like pair counts and rerun resolution to Results.

2. Right now the animation focused too much on the part that train and val accuracy plateau. I want to focus on the period between the training starts and the training statrts to plateau for hundres of steps. Can we use a linear time scale?

3. **Break up overloaded sentences.**  
   Many sentences carry several claims and parenthetical qualifications—for example lines 63–71 and 237–243. Aim for one main claim per sentence and move validation details into separate paragraphs.

4. **Reduce terminology and define a smaller core vocabulary.**  
   Readers encounter “relative endpoint distance,” “plateau fraction,” “plateau contrast,” “stable-region count,” “norm-rescaled SLERP,” and more. Clearly distinguish the primary metric from secondary controls, ideally with a short “How to read the plots” box.

5. **Remove repetition and let figures carry evidence.**  
   “Gradual emergence,” “late boundary movement,” and “after accuracy saturates” recur in the Summary, Results, captions, and Conclusion. State each finding fully once; elsewhere use short references. The image captions, especially lines 190, 194, 202, and 219, can also be substantially shortened.

6. How is accuracy/confidence calculated?

7. It is uncleared which layer'd you are showing in the plots

8. Can you also show train and val loss in the animation?