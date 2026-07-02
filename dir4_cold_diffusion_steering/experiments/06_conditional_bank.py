"""Iteration 6 — DIRECTION-CONDITIONAL corrector trained on a VECTOR BANK (S4c).

Exp 5 found the single-vector corrector r_θ(h, z, α) is direction-SPECIFIC: trained on
sentiment it gave ≈0% fluency recovery on a held-out formality vector, because it never
sees v (only implicitly through z) and only ever saw one direction. The proposal's direct
fix (Failure Mode 4 mitigation): make the corrector CONDITIONAL on the steering direction,
r_θ(h, z, v̂, α), and TRAIN it on a small BANK of directions. Then ask:

    Does ONE conditional corrector, trained on a bank of directions, (a) correct every
    IN-BANK direction at once, and (b) TRANSFER to a HELD-OUT direction it never trained on
    — unlike the Exp-5 single-vector corrector, whose transfer was ≈0%?

Bank (3 training directions): sentiment v1, formality v2, concreteness v3.
Held-out (never trained):     certainty v4.
All are DiffMean vectors at resid_post block 6; we report their pairwise cosines.

Parameterization is unchanged and still projection-preserving by construction:
    ĥ = z + P_{v⊥} r_θ(h, z, v̂, α),   z = h + α v.
The ONLY change vs Exp 3/5 is that r_θ now takes v̂ as an extra input and training samples
a (direction, α) pair per step. Native single-direction oracles (same conditional arch,
trained on one direction) give the per-vector ceiling.

Model: GPT-2 small, resid_post block 6. Same held-out 100 FineWeb docs as Exp 3/4/5.
Outputs: results/06_conditional_bank.json + plots/06_conditional_bank.png.
"""
import os, json, importlib.util
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import resid_post, fineweb_texts, load_model, DATA, DEVICE
from projections import normalize_vector

HERE = os.path.dirname(__file__)
PLOTS = os.path.join(HERE, "..", "plots")
RESULTS = os.path.join(HERE, "..", "results")

# reuse Exp 3 machinery verbatim (identical LM-loss / hook / eval path)
_spec = importlib.util.spec_from_file_location("exp03", os.path.join(HERE, "03_learned_corrector.py"))
exp03 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exp03)
FuncPatcher = exp03.FuncPatcher
batched_ids = exp03.batched_ids
lm_loss_fn = exp03.lm_loss_fn
gaussian_stats = exp03.gaussian_stats
mahalanobis = exp03.mahalanobis
LAYER = exp03.LAYER
# diffmean + formality pairs from Exp 5
_spec5 = importlib.util.spec_from_file_location("exp05", os.path.join(HERE, "05_heldout_vector.py"))
exp05 = importlib.util.module_from_spec(_spec5); _spec5.loader.exec_module(exp05)
diffmean_vector = exp05.diffmean_vector

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

# ---- new concept: concreteness (concrete/sensory ↔ abstract/conceptual) ----
CONCRETE = [
    "The rusty iron gate creaked as she pushed it open with both hands.",
    "He poured steaming coffee into a chipped blue mug on the table.",
    "The wet dog shook water all over the muddy kitchen floor.",
    "She stacked the heavy cardboard boxes against the brick wall.",
    "A red bicycle leaned against the peeling white fence outside.",
    "The chef sliced the ripe tomatoes on a wooden cutting board.",
    "Rain drummed loudly on the tin roof of the old barn.",
    "He tightened the loose screw with a small rusty screwdriver.",
    "The cat curled up on the warm windowsill in the sunlight.",
    "She tied the frayed shoelaces on her scuffed leather boots.",
    "Smoke drifted from the crackling campfire into the dark trees.",
    "The waiter carried a tray of clinking glasses across the room.",
    "Cold wind rattled the bare branches against the window pane.",
    "He hammered the bent nail into the splintered wooden plank.",
    "The baker dusted flour over the sticky dough on the counter.",
    "Muddy footprints tracked across the freshly mopped tile floor.",
]
ABSTRACT = [
    "Justice requires a careful balance between mercy and accountability.",
    "Freedom means little without the responsibility that sustains it.",
    "The concept of time shapes how we interpret memory and hope.",
    "Wisdom emerges slowly from the accumulation of considered experience.",
    "Democracy depends on the trust citizens place in shared institutions.",
    "Meaning is constructed through the relationships between ideas.",
    "Progress often demands sacrificing comfort for uncertain ideals.",
    "Identity is a fluid negotiation between memory and aspiration.",
    "Truth resists simplification into convenient absolute categories.",
    "Morality evolves as societies reconsider their inherited assumptions.",
    "Knowledge without humility tends toward dangerous overconfidence.",
    "Equality is an aspiration continually redefined by each generation.",
    "Creativity arises from the tension between constraint and freedom.",
    "Power corrupts when it escapes meaningful accountability.",
    "Purpose gives coherence to an otherwise fragmented existence.",
    "Belief and doubt coexist within any honest understanding.",
]

# ---- held-out concept: certainty (assertive ↔ hedged) ----
CERTAIN = [
    "This is definitely the correct answer, without any doubt whatsoever.",
    "The results prove conclusively that the treatment works every time.",
    "I am absolutely certain that the meeting starts at nine sharp.",
    "There is no question that this approach will succeed completely.",
    "The evidence clearly establishes the defendant's guilt beyond doubt.",
    "This method always produces the same reliable outcome, guaranteed.",
    "We know for a fact that the shipment arrived yesterday morning.",
    "It is undeniable that the policy improved every measured indicator.",
    "The data definitively confirms our original hypothesis was right.",
    "You will certainly pass the exam if you follow these steps.",
    "The bridge is completely safe and will never collapse under load.",
    "History shows unmistakably that such regimes always eventually fall.",
    "I guarantee the product will exceed all of your expectations.",
    "The theorem is proven true for absolutely all possible cases.",
    "This is precisely how the mechanism works, exactly as described.",
    "The forecast is certain: it will rain heavily tomorrow afternoon.",
]
HEDGED = [
    "This might possibly be the right answer, though I'm not entirely sure.",
    "The results seem to suggest the treatment could perhaps help somewhat.",
    "I think the meeting maybe starts around nine, but don't quote me.",
    "It's conceivable this approach may work, under certain conditions perhaps.",
    "The evidence arguably points toward guilt, though doubts remain.",
    "This method sometimes produces roughly similar outcomes, more or less.",
    "It appears the shipment might have arrived, possibly yesterday.",
    "The policy seemingly improved a few indicators, if I recall correctly.",
    "The data could tentatively support our hypothesis, to some degree.",
    "You might perhaps pass the exam, assuming things go reasonably well.",
    "The bridge is probably fairly safe under most ordinary conditions.",
    "History perhaps hints that such regimes may sometimes eventually fall.",
    "I suppose the product could maybe meet some of your expectations.",
    "The theorem seems to hold in many cases, though not certainly all.",
    "This is roughly how the mechanism might work, as far as I can tell.",
    "The forecast is uncertain; it may possibly rain tomorrow, or not.",
]


class CondCorrector(nn.Module):
    """r_θ(h, z, v̂, α): 4-layer MLP conditioned on the (normalized) steering direction v̂.
    Last layer zero-init so it starts as the identity (ĥ = z = raw steering)."""
    def __init__(self, d=768, hidden=1024, n_layers=4):
        super().__init__()
        dims = [3 * d + 1] + [hidden] * (n_layers - 1) + [d]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, z, vhat, alpha):
        a = torch.full_like(h[..., :1], alpha / 8.0)
        vh = vhat.expand_as(h)                       # broadcast [d] -> [...,d]
        x = torch.cat([h / 100.0, z / 100.0, vh, a], dim=-1)
        return self.net(x)


def make_hat_cond(corr, h, alpha, vt, vhat_t):
    """ĥ = z + P_{v⊥} r_θ(h, z, v̂, α)."""
    z = h + alpha * vt
    r = corr(h, z, vhat_t, alpha)
    r_perp = r - (r @ vhat_t).unsqueeze(-1) * vhat_t
    return z + r_perp, r_perp


def train_cond(corr, bank, train_texts, seq_len=64, batch=8, epochs=8, lr=1e-3,
               lam_near=0.05, amin=0.5, amax=8.0):
    """Train the conditional corrector, sampling (direction, α) per step from `bank`
    (list of (vt, vhat_t)). Identical LM-loss objective / hook as Exp 3."""
    model, tok = load_model()
    opt = torch.optim.Adam(corr.parameters(), lr=lr)
    rng = np.random.RandomState(SEED)
    corr.train(); step = 0
    for ep in range(epochs):
        order = rng.permutation(len(train_texts))
        texts = [train_texts[i] for i in order]
        for ids, mask in batched_ids(tok, texts, seq_len, batch):
            alpha = float(rng.uniform(amin, amax))
            vt, vhat_t = bank[rng.randint(len(bank))]
            store = {}

            def fn(h):
                h = h.detach()
                hat, r_perp = make_hat_cond(corr, h, alpha, vt, vhat_t)
                store["rp"] = r_perp
                return hat

            with FuncPatcher(model, LAYER, fn):
                logits = model(ids, attention_mask=mask).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = ids[:, 1:]; m = mask[:, 1:].bool()
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lm = nll[m].mean()
            near = (store["rp"] / 100.0).pow(2).sum(-1).mean()
            loss = lm + lam_near * near
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step % 40 == 0:
                print(f"  ep{ep} step{step:4d} a={alpha:4.1f} LM={lm.item():.3f}", flush=True)
    corr.eval(); return corr


@torch.no_grad()
def cond_acts(corr, Hs, alpha, vt, vhat_t, batch=4096):
    out = []
    for i in range(0, len(Hs), batch):
        h = torch.tensor(Hs[i:i + batch], device=DEVICE)
        hat, _ = make_hat_cond(corr, h, alpha, vt, vhat_t)
        out.append(hat.cpu().numpy())
    return np.concatenate(out, 0)


def main():
    model, tok = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    print("device:", DEVICE, flush=True)

    # ---- build the four DiffMean directions ----
    v1 = np.load(os.path.join(DATA, "sentiment_vec_layer6.npz"))["v"].astype(np.float32)
    v2 = np.load(os.path.join(DATA, "formality_vec_layer6.npy")).astype(np.float32)
    v3 = diffmean_vector(CONCRETE, ABSTRACT)
    v4 = diffmean_vector(CERTAIN, HEDGED)                     # held-out
    np.save(os.path.join(DATA, "concreteness_vec_layer6.npy"), v3)
    np.save(os.path.join(DATA, "certainty_vec_layer6.npy"), v4)

    names = ["sentiment", "formality", "concreteness", "certainty"]
    vecs = {"sentiment": v1, "formality": v2, "concreteness": v3, "certainty": v4}
    heldout = "certainty"
    bank_names = ["sentiment", "formality", "concreteness"]

    def T(v):
        vhat = normalize_vector(v)
        return (torch.tensor(v, device=DEVICE), torch.tensor(vhat, device=DEVICE), vhat)

    tens = {n: T(vecs[n]) for n in names}
    norms = {n: float(np.linalg.norm(vecs[n])) for n in names}
    # pairwise cosines (report independence of the directions)
    cos = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cos[f"{a}|{b}"] = float(vecs[a] @ vecs[b] /
                                    (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b])))
    print("norms:", {n: round(norms[n], 1) for n in names}, flush=True)
    print("cosines:", {k: round(v, 3) for k, v in cos.items()}, flush=True)

    # ---- clean Gaussian + eval/train split (identical protocol to Exp 3/4/5) ----
    H = resid_post(fineweb_texts(400), LAYER, seq_len=128, batch=16)
    mu, cov, prec = gaussian_stats(H)
    hnorm = float(np.linalg.norm(H, axis=1).mean())
    idx = np.random.RandomState(0).choice(len(H), size=min(20000, len(H)), replace=False)
    Hs = H[idx]
    dm_clean = float(mahalanobis(Hs, mu, prec).mean())
    print(f"clean D_M={dm_clean:.2f} mean|h|={hnorm:.2f}", flush=True)

    all_texts = fineweb_texts(800)
    train_texts = all_texts[500:800]
    eval_texts = all_texts[400:500]

    # ---- train ONE conditional corrector on the 3-direction bank ----
    bank = [(tens[n][0], tens[n][1]) for n in bank_names]
    print(f"training BANK conditional corrector on {bank_names} ...", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    corr_bank = CondCorrector().to(DEVICE)
    print(f"cond corrector params: {sum(p.numel() for p in corr_bank.parameters())/1e6:.2f}M",
          flush=True)
    train_cond(corr_bank, bank, train_texts)

    # ---- native oracle on the held-out direction (same arch, single direction) ----
    print(f"training NATIVE oracle on held-out '{heldout}' ...", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    corr_native = CondCorrector().to(DEVICE)
    train_cond(corr_native, [(tens[heldout][0], tens[heldout][1])], train_texts)

    # ---- eval ----
    alphas = [1.0, 2.0, 4.0, 6.0, 8.0]
    clean_loss = lm_loss_fn(eval_texts, LAYER, lambda h: h)
    print(f"clean eval loss = {clean_loss:.3f}", flush=True)

    # per direction: raw + bank(one model) ; held-out also gets native oracle
    out = {n: {"raw_dlm": [], "bank_dlm": [], "bank_dm": [], "raw_dm": [],
               "retention": []} for n in names}
    out[heldout]["native_dlm"] = []
    out[heldout]["native_dm"] = []

    for n in names:
        vt, vhat_t, vhat = tens[n]
        for a in alphas:
            def f_raw(h):  return h + a * vt
            def f_bank(h):
                hat, _ = make_hat_cond(corr_bank, h, a, vt, vhat_t); return hat
            out[n]["raw_dlm"].append(lm_loss_fn(eval_texts, LAYER, f_raw) - clean_loss)
            out[n]["bank_dlm"].append(lm_loss_fn(eval_texts, LAYER, f_bank) - clean_loss)
            Z = Hs + a * vecs[n][None, :]
            Zb = cond_acts(corr_bank, Hs, a, vt, vhat_t)
            out[n]["raw_dm"].append(float(mahalanobis(Z, mu, prec).mean()))
            out[n]["bank_dm"].append(float(mahalanobis(Zb, mu, prec).mean()))
            out[n]["retention"].append(float(((Zb - Hs) @ vhat).mean()))
            if n == heldout:
                def f_na(h):
                    hat, _ = make_hat_cond(corr_native, h, a, vt, vhat_t); return hat
                out[n]["native_dlm"].append(lm_loss_fn(eval_texts, LAYER, f_na) - clean_loss)
                Zn = cond_acts(corr_native, Hs, a, vt, vhat_t)
                out[n]["native_dm"].append(float(mahalanobis(Zn, mu, prec).mean()))
        tag = "HELD-OUT" if n == heldout else "in-bank"
        print(f"[{tag:8}] {n:12} α=8  raw={out[n]['raw_dlm'][-1]:+.3f} "
              f"bank={out[n]['bank_dlm'][-1]:+.3f}"
              + (f" native={out[n]['native_dlm'][-1]:+.3f}" if n == heldout else ""),
              flush=True)

    res = dict(layer=LAYER, norms=norms, cosines=cos, dm_clean=dm_clean,
               clean_loss=clean_loss, alphas=alphas, bank=bank_names, heldout=heldout,
               results=out, n_train_texts=len(train_texts), n_eval_texts=len(eval_texts))
    with open(os.path.join(RESULTS, "06_conditional_bank.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figure: (a) in-bank recovery bars at α=8 ; (b) held-out ΔLM sweep ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # panel (a): recovery% at α=8 for every direction under the SINGLE bank model
    rec8 = [100.0 * (out[n]["raw_dlm"][-1] - out[n]["bank_dlm"][-1]) / out[n]["raw_dlm"][-1]
            for n in names]
    colors = ["#2980b9" if n in bank_names else "#e67e22" for n in names]
    ax[0].bar(range(len(names)), rec8, color=colors)
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([f"{n}\n({'bank' if n in bank_names else 'HELD-OUT'})"
                           for n in names], fontsize=9)
    ax[0].set_ylabel("fluency recovery at α=8 (%)")
    ax[0].set_title("(a) ONE conditional corrector, per-direction recovery")
    for i, r in enumerate(rec8):
        ax[0].text(i, r + (2 if r >= 0 else -6), f"{r:.0f}%", ha="center", fontsize=9)
    # panel (b): held-out certainty ΔLM sweep — raw / bank(transfer) / native
    n = heldout
    ax[1].plot(alphas, out[n]["raw_dlm"], "o-", color="#c0392b", label="raw steer  h+αv")
    ax[1].plot(alphas, out[n]["bank_dlm"], "o-", color="#e67e22",
               label="bank corrector (transfer to held-out)")
    ax[1].plot(alphas, out[n]["native_dlm"], "o-", color="#27ae60",
               label="native corrector (oracle)")
    ax[1].axhline(0, ls="--", color="k", lw=1)
    ax[1].set_xlabel(r"steering strength $\alpha$")
    ax[1].set_ylabel(r"$\Delta$ LM loss (nats)")
    ax[1].set_title(f"(b) held-out direction '{heldout}' (lower is better)")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Direction-conditional corrector on a vector bank: one model corrects every "
                 "in-bank\ndirection and partially transfers to a held-out one "
                 "(GPT-2 small, layer 6)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "06_conditional_bank.png"), dpi=120)
    plt.close(fig)
    print("saved figure + results", flush=True)


if __name__ == "__main__":
    main()
