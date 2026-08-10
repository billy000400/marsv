# Does an activation plateau hold information the model has not used yet?

## Summary

A language model's hidden state changes smoothly as you smoothly change its input — except that
sometimes it doesn't. Interpolating between two inputs often produces an **activation plateau**: the
representation sits still over a wide range of the interpolation, then swings across quickly. If we
want to audit what a model is "about to do", it matters whether a plateau organizes only the token
the model outputs *right now*, or also information it will need *later*.

We tested this with a designed example: a prompt whose two endpoints make the *same* immediate
prediction but should diverge one token later. The example failed its own precondition — GPT-2 Large
does not perform the codebook lookup the design assumed, so the verdict on the planned test is
**invalid example** (conclusion 3 of the pre-registered three). The measurement that does survive is
about propagation: with the interpolated embedding injected at the symbol position, the logit vector
read out one position downstream — at a token that can reach the symbol only through attention — is
still plateau-shaped, with transition width 0.38 against 0.80 for a linear response, and its
transition sits at the same interpolation position as the immediate one. The information is
attenuated roughly fourfold in the process and never changes the model's actual output token. So
plateau structure does travel forward through the network's own routing, but this example gives no
evidence that it organizes *behaviourally relevant* future information.

## Methods

### Data & Model

Pretrained **GPT-2 Large** (774M parameters, 36 transformer blocks), evaluation mode, float32, greedy
readout with no sampling. No training or fine-tuning of any kind. There is no dataset: the experiment
is one hand-designed prompt, run at 101 interpolation positions.

The prompt is a codebook instruction followed by a symbol:

```text
prefix P : "Use the codebook A = cat and B = dog. Complete: Symbol"
endpoint A : " A"      endpoint B : " B"      shared successor S : " means"
```

` A`, ` B`, ` means`, ` cat` and ` dog` are each a single GPT-2 token, and appending them does not
retokenize the prefix (both verified in code). The symbol occupies position 15.

Two sequences are run at every interpolation position, differing only in whether the shared successor
is appended:

- **immediate readout** — `P + [A→B]`, next-token logits at the symbol position;
- **delayed readout** — `P + [A→B] + S`, next-token logits at the ` means` position.

**Hook point.** The interpolated vector replaces the *input embedding* at position 15, before block 0,
so it is fed through the entire network. Position embeddings and all other tokens are untouched. This
placement is what makes the delayed readout meaningful: at the ` means` position the network has no
direct copy of the symbol embedding, so anything it knows about the interpolation must arrive through
attention across positions.

### Metrics

**Interpolation.** We need a path between the two token embeddings that keeps vector length in a range
the model actually sees; naive averaging shrinks the norm in the middle and would create an artificial
"nothing here" region that could masquerade as a plateau. We therefore use the norm-corrected SLERP of
Matthew's activation-plateau experiment — spherical interpolation of the direction with a linear
interpolation of the norm. With $e_A, e_B$ the token embeddings, $u = e_A/\lVert e_A\rVert$,
$v = e_B/\lVert e_B\rVert$ and $\Omega = \arccos(u\cdot v)$:

```math
e(t) = \big[(1-t)\lVert e_A\rVert + t\lVert e_B\rVert\big]\,
       \frac{\sin\!\big((1-t)\Omega\big)\,u + \sin\!\big(t\Omega\big)\,v}{\sin \Omega},
\qquad t \in \{0, 0.01, \dots, 1\}.
```

Here $\Omega = 1.227$ rad ($\cos = 0.337$), so the two embeddings are far from parallel and the
spherical path is genuinely curved.

**Endpoint validity.** The design only makes sense if both endpoints predict the same next token and
different tokens after the successor. We check the top-1 token of all four endpoint sequences against
the plan, and quantify how different two distributions are with the **Jensen–Shannon divergence**
(JSD), a symmetric, bounded measure of distributional distance in nats — 0 means identical,
larger means more different:

```math
\mathrm{JSD}(p\,\Vert\,q) = \tfrac{1}{2}\mathrm{KL}(p\,\Vert\,m) + \tfrac{1}{2}\mathrm{KL}(q\,\Vert\,m),
\qquad m = \tfrac{1}{2}(p+q).
```

This decides the verdict, and is what Figure 1 reports.

**Relative logit distance.** To ask *where along the interpolation* the output moves, we need a
quantity that is 0 at one endpoint, 1 at the other, and insensitive to the overall size of the logit
change — otherwise a readout with a small absolute swing would look flat for trivial reasons. With
$z(t)$ the logit vector at interpolation position $t$ and $z_A = z(0)$, $z_B = z(1)$:

```math
d(t) = \frac{\lVert z(t) - z_A\rVert_2}{\lVert z(t) - z_A\rVert_2 + \lVert z(t) - z_B\rVert_2}.
```

Read it as "what fraction of the way from A to B is the output at this point". Figure 2 plots it.

**Transition width.** The plateau claim is about *shape*, so we summarise each curve by how much of
the interpolation the crossing occupies. With $t_q$ the first interpolation position where
$d(t) \ge q$:

```math
w = t_{0.9} - t_{0.1}.
```

Small $w$ means flat regions at both ends and a fast swing between them — a plateau. We also report
the midpoint $t_{0.5}$, which says *where* the swing happens; two readouts whose midpoints coincide
are crossing the same boundary.

**Endpoint separation.** Because $d(t)$ is scale-free, it can make a negligible difference look like a
dramatic transition. We report the absolute size of the gap being normalised, $\lVert z_A - z_B\rVert_2$,
so the delayed curve can be read against how much signal actually reaches that position.

**Top-2 margin.** A change in the logit vector matters behaviourally only if it can change the output.
At the delayed readout we track the gap between the largest and second-largest logit; a margin that
never approaches 0 means no interpolation position comes close to flipping the prediction.

### Baselines

**Linear reference.** A model whose output moved uniformly with the input would trace $d(t) = t$,
giving

```math
w_{\mathrm{lin}} = t_{0.9} - t_{0.1} = 0.9 - 0.1 = 0.8 .
```

This is the "no plateau" null shape, drawn as the dotted line in Figure 2, and $w$ below roughly 0.5
was pre-registered as evidence of a plateau.

**Immediate readout as reference for the delayed one.** The immediate readout sees the interpolated
embedding directly, so its plateau is the strongest form the effect can take in this prompt. Comparing
the delayed curve against it separates "the plateau propagated" from "the plateau was created anew
downstream": equal midpoints indicate the same boundary, and the change in $w$ measures how much
propagation blurs it.

## Results

### The designed example does not work in GPT-2 Large

The experiment is pre-registered to stop at conclusion 3 if the endpoints misbehave, and they do. After
` A` the model's top-1 token is ` =` with probability 0.340 (after ` B`, ` =` with 0.525); the planned
successor ` means` gets 6.68e-4 and 4.50e-4 — three orders of magnitude below. GPT-2 Large reads the
prompt as continuing the codebook listing, so the two endpoints agree on the next token for the wrong
reason. After ` A means` and ` B means` the top-1 is a quote mark in both cases; ` cat` gets 0.061 and
0.046, ` dog` 0.010 and 0.011. The intended contrast is present only as a faint preference in the
right direction, and it is swamped: endpoint JSD at the delayed readout is 0.0115 nats, seven times
smaller than the 0.0861 nats the two endpoints already differ by *before* the successor. Figure 1
shows the immediate failure across the whole interpolation.

![Probability of the top-1 token and of ' means' across the interpolation](plots/immediate_readout.png)

**Figure 1.** The planned successor is never a live option, at any interpolation position. x:
interpolation position `t` (0 = ` A`, 1 = ` B`); y: probability, log scale. Solid = probability of the
model's own top-1 token (` =` at every `t`); dashed = probability of ` means`. The gap is roughly
three orders of magnitude, so no choice of readout threshold rescues the design.

Following the pre-registered rule, we do not adjust the prompt to make the example work. The planned
delayed-plateau question — does the output diverge sharply *after* a shared successor — is therefore
unanswered for this prompt, and we make no claim about it.

### Plateau shape survives one token of propagation

The interpolation sweep itself remains interpretable, and it answers a narrower question that the
failed example does not touch: when the interpolated embedding is injected at the symbol, does the
plateau structure still exist at a *different* position, reachable only through attention? It does.
Figure 2 shows the delayed curve is flat near both endpoints and crosses quickly in the middle, with
width $w = 0.38$ against the linear null's 0.80 — comfortably inside the pre-registered $w < 0.5$
criterion, and monotone at every step.

![Relative logit distance versus interpolation position for both readouts](plots/delayed_distance.png)

**Figure 2.** Both readouts are plateau-shaped, and they cross at the same place. x: interpolation
position `t` (0 = ` A`, 1 = ` B`); y: relative logit distance `d(t)`, 0 at the ` A` endpoint and 1 at
the ` B` endpoint. Solid = immediate readout ($w = 0.27$), dashed = delayed readout after ` means`
($w = 0.38$), dotted = linear reference $d = t$ ($w = 0.80$). Thin horizontal lines mark the 0.1 and
0.9 levels defining $w$.

The two midpoints agree closely — $t_{0.5} = 0.45$ immediate, $0.42$ delayed — which is the
informative part. The downstream position is not forming its own boundary somewhere else; it is
responding to the same switch in the symbol's representation, one token later and through attention
only. Propagation does cost signal and sharpness: the endpoint separation drops from
$\lVert z_A - z_B\rVert_2 = 300.2$ at the symbol to $75.4$ after ` means`, a factor of 4.0, and the
transition broadens from $w = 0.27$ to $0.38$, moving about 15% of the way toward the linear null.
The practical reading is that a plateau detected at one position is not a purely local artifact of the
readout there — it is a property of a representation that keeps its shape while the network routes it
forward. For interpretability work that reads plateaus off a single position, that is mild good news:
the boundary is not an artifact of where you happened to look.

### The downstream divergence never reaches the output

The scale caveats matter, and Figure 3 makes the strongest one concrete. Across the whole
interpolation the delayed top-1 token is a quote mark, and the gap between the top two logits stays in
[0.43, 0.69] — never near 0, so no interpolation position comes close to flipping the prediction.
p(` cat`) leads p(` dog`) everywhere, from a ratio of 6.0 at `t = 0` to 4.0 at `t = 1`.

![Probability of ' cat' and ' dog' at the delayed readout across the interpolation](plots/delayed_tokens.png)

**Figure 3.** The intended semantic flip never happens. x: interpolation position `t`; y: probability
at the delayed readout. Solid = p(` cat`), dashed = p(` dog`). The curves move slightly in the
expected directions but never cross, so the codebook lookup the design assumed is absent.

The delayed plateau in Figure 2 is thus a statement about the geometry of the logit vector, not about
behaviour. The model's future output is unchanged by an input edit that its downstream representation
clearly registers.

## Conclusion

For this prompt, GPT-2 Large does not implement the codebook, so the delayed-plateau test as designed
returns **conclusion 3: invalid example**. We report that as the verdict and do not generalise from
the secondary measurement.

That secondary measurement is still worth stating: plateau structure injected at one token is
recoverable at a later token that sees it only through attention, with the transition in the same
place ($t_{0.5}$ 0.45 vs 0.42), a modestly wider crossing ($w$ 0.27 vs 0.38) and roughly fourfold
attenuation of the endpoint gap (300.2 vs 75.4). Limitations are severe and worth naming plainly:
this is a single prompt, a single model, one injection site and one downstream position; there is no
control prompt and no statistical test, so the width values describe this curve and nothing more.
Most importantly, the downstream change never crosses a decision boundary, so nothing here shows that
plateaus organize information the model will actually *act* on later. Testing that claim needs a
prompt where the model demonstrably performs the delayed lookup — establishing which prompts those
are, by checking endpoint behaviour first, is the prerequisite this iteration turned up.
