Please write a reproducible Python script that reads the existing results and generates one summary figure as output.
Goal
Create a 2×3 figure layout summarizing the relationship between interpolation distance and transition width across models.
Panels 1–5: scatter plots
Make scatter plots for these five models:
1. GPT-2 Large
2. GPT-2 XL
3. Pythia 1.4B
4. Llama 3.1-8B
5. Qwen3-8B
For each of these five panels:
- Only use final-layer results.
- Only keep monotonic curves.
- Plot:
  - x-axis: interpolation distance
  - y-axis: transition width
- Use a shared y-axis range across all five scatter plots so the transition widths are directly comparable.
- Let each model keep its own x-axis range (do not force a shared x-axis range if the distance scales differ across models).
- Use a scatter plot style that is easy to read:
  - small-to-medium point size
  - some transparency (alpha) to reduce overplotting
  - no unnecessary clutter
- Compute the Pearson correlation coefficient between:
  - interpolation distance
  - transition width
    using only the monotonic curves for that model.
- Show that Pearson correlation clearly in each subplot title, for example:
  - GPT-2 XL (Pearson r = 0.03)
Panel 6: summary correlation panel
Use the sixth panel (bottom-right) as a compact summary plot.
- y-axis: model name
- x-axis: Spearman correlation coefficient
- Compute one Spearman coefficient per model, again using:
  - interpolation distance
  - transition width
  - only monotonic curves
  - final layer only
- Set the x-axis range to [-1, 1]
- Plot one point per model
- Draw vertical reference lines at:
  - x = -1
  - x = 1
- Annotate these lines so the plot makes clear what they mean, e.g.:
  - -1 = perfect negative monotonic relationship
  - 1 = perfect positive monotonic relationship
- Also draw a vertical line at x = 0 if helpful, since that improves readability.
Important implementation requirements
Please make the script:
1. Reproducible
   - It should run from the command line.
   - It should read the existing results from disk.
   - It should save the final figure to disk.
2. Readable and easy to fine-tune
   - Organize the code into clear sections and helper functions.
   - Put all plotting style knobs near the top of the file as configuration variables, such as:
     - figure size
     - font sizes
     - marker size
     - alpha
     - line widths
     - subplot spacing
     - output DPI
   - Add comments explaining where to quickly modify:
     - filtering logic
     - panel titles
     - axis labels
     - y-axis sharing
     - colors / marker styles
     - summary-panel annotations
3. Robust
   - The script should clearly define what counts as:
     - “final layer”
     - “monotonic curve”
     - “transition width”
     - “interpolation distance”
   - If the existing result format requires assumptions, isolate those assumptions in one place in the code and document them clearly.
   - If a model has no qualifying monotonic samples after filtering, handle that gracefully.
Expected output
The script should produce:
- a single publication-quality figure file (e.g. .png and/or .pdf)
- a 2×3 layout:
  - first 5 panels = scatter plots
  - final panel = Spearman summary across models
Suggested code structure
Please structure the script roughly like this:
- imports
- user-editable configuration
- data loading functions
- filtering / metric extraction functions
- correlation computation functions
- plotting helper functions
- main function
- CLI entry point
Extra request
Please include brief comments in the script explaining where I should edit things if I want to fine-tune the plotting style later.

Data source
Use only data that already exists in this direction; do not run any new model sweeps.
Read the current block-0 result set: results/block0_resid_post/sweep.csv (GPT-2 Large)
and results/block0_resid_post/gen/<model>.csv (the other four models).
Do not use the older input-embedding result set (results/sweep.csv, results/gen/).
