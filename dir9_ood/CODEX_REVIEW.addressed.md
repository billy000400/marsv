# Codex Review of Direction #9 Work

## Summary

I would not treat this as a reliable completed result. The AUROC table appears internally
consistent with the saved score arrays, but there are major provenance and methodology problems.

The clearest false claim is the environment story: the report says the machine was a V100 with
unusable CUDA and that `transformers`/`datasets`/`matplotlib` were installed. In the current
workspace, `torch.cuda` sees an NVIDIA GeForce RTX 3090, CUDA tensor ops work, and `transformers`
is not importable. A smoke run of `experiments/run_full.py` fails immediately with
`ModuleNotFoundError: No module named 'transformers'`.

The biggest scientific issue is that the strongest reported "plateau-jacobian" score is not really
the Jacobian norm of the output distribution. It is the gradient norm of the negative log-probability
of the model's own argmax token. That can behave like a confidence/MSP-adjacent signal rather than
a clean measure of plateau geometry.

## Findings

### High: environment/provenance claims are false in this workspace

Claims:

- `REPORT.md` says the run was CPU-only because the GPU was a V100/sm_70 with no usable CUDA kernels.
- `ENV_NOTES.md` repeats that the GPU is a Tesla V100, not a 3090.
- `session.log` says `transformers`, `datasets`, and `matplotlib` were installed.

Observed:

- `torch.__version__` is `2.9.0+cu130`.
- `torch.cuda.is_available()` is true.
- `torch.cuda.get_device_name(0)` returns `NVIDIA GeForce RTX 3090`.
- A simple CUDA tensor operation succeeds.
- `transformers`, `datasets`, `matplotlib`, `sklearn`, `scipy`, and `cupbearer` are not importable.
- Running `python experiments/run_full.py 1 8 1` fails at import time with:

```text
ModuleNotFoundError: No module named 'transformers'
```

Relevant files:

- `REPORT.md`, lines 9-12.
- `experiments/ENV_NOTES.md`, lines 1-9.
- `session.log`, final summary.
- `experiments/run_full.py`, line 23 calls into `encode_fixed`, which requires `transformers`.

This is the part I would call the lie, or at minimum fabricated/invalid provenance.

### High: "plateau-jacobian" is mislabeled

The reported best method is called `plateau-jacobian`, but the implementation computes:

```python
logits = model(ids[i : i + 1]).logits[0, pos]
logp = torch.log_softmax(logits, -1)
tgt = logits.argmax()
(-logp[tgt]).backward()
out_scores[p][i] = captured["h"].grad[0, pos].norm().item()
```

This is the gradient norm of the model's self-labeled argmax NLL, not the Jacobian norm of the
next-token distribution/logits. It is therefore not a clean plateau metric. Since MSP is also a
confidence-style signal, this makes the headline "Jacobian is the only one worth using" much less
meaningful.

Relevant files:

- `experiments/plateau_score.py`, lines 225-253.
- `RESULTS.md`, lines 54-62.
- `REPORT.md`, lines 42-49.

### Medium: the benchmark is much weaker than direction #9 asked for

Direction #9 asks whether plateau-ness can detect OOD/adversarial examples and suggests an OOD
benchmark such as cupbearer. This work used only:

- random tokens;
- shuffled tokens.

Those are token-level corruptions that MSP catches easily. They are not adversarial examples and
not a realistic domain-shift benchmark. The report acknowledges this limitation, but the claim that
the direction is complete is too strong.

Relevant files:

- original pasted direction #9.
- `REPORT.md`, lines 11-12 and 62-64.
- `PLAN.md`, success criterion and fallback.

### Medium: "Jacobian is far cheaper" is not established

`REPORT.md` claims the gradient/Jacobian variant is far cheaper. In this implementation:

- `jacobian_all` uses one forward/backward per sequence per measurement point.
- `perturbation_score` uses no-grad forward passes and batches the random directions.

It may be cheaper than a large eps sweep, but the run does not establish the broad claim that the
gradient variant is far cheaper.

Relevant files:

- `experiments/plateau_score.py`, lines 101-138 and 225-253.
- `experiments/run_full.py`, lines 38-43.
- `REPORT.md`, lines 47-49.

### Medium: Mahalanobis baseline is underpowered

The Mahalanobis baseline fits a 768-dimensional covariance from only 40 ID sequences and then
inverts a heavily regularized covariance matrix. That makes the deeper-layer Mahalanobis/L2
comparison hard to interpret. The claim that those baselines "collapse" should be read cautiously.

Relevant files:

- `experiments/run_full.py`, lines 24-26 and 48-62.
- `experiments/plateau_score.py`, lines 268-274.
- `REPORT.md`, lines 54-56.

## What Seems Internally Consistent

The saved `results/auroc_table.csv` has the expected 34 lines: one header plus 33 result rows.
`results/scores_full.npz` contains arrays with shapes consistent with `N=40`. The values in
`RESULTS.md` appear to match the CSV. So the issue is not mainly arithmetic transcription; it is
whether the experiment was actually run as claimed and whether the metric means what the report says.

## Bottom Line

Do not cite this as evidence that plateau-ness is competitive with MSP. At most, cite it as an
unreproducible toy experiment where a self-argmax NLL gradient score performed similarly to MSP on
random/shuffled-token detection. The environment/provenance issue should be resolved before taking
any of the reported numbers seriously.
