"""S1: ' big' -> 20 diverse tokens. Plot y(t) at blocks 18 and 35 with t_0.1/t_0.9 marks."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CVD, PLOTS, RESULTS, TS, Runner, load, width, y_curve)

WORDS = [" small", " large", " huge", " tiny", " red", " old", " empty", " the", " and", " dog",
         " quickly", " sold", " 1995", ".", ",", "ing", " Paris", " é", "!!", " bigger"]


def main():
    tok, m, dev = load()
    r = Runner(tok, m, dev)
    assert tok.decode([r.id_a]) == " big", tok.decode(r.ids)
    print("prompt ids", r.ids, [tok.decode([i]) for i in r.ids])
    ids_b = [tok.encode(w)[0] for w in WORDS]
    assert all(len(tok.encode(w)) == 1 for w in WORDS), [w for w in WORDS if len(tok.encode(w)) != 1]
    t = torch.tensor(TS, device=dev, dtype=torch.float32).unsqueeze(1)
    ea = r.wte[r.id_a]
    rows, curves = [], {}
    for w, ib in zip(WORDS, ids_b):
        eb = r.wte[ib]
        mid, fin = r.run((1 - t) * ea + t * eb)
        # check that endpoint runs reproduce direct token runs
        ym, yf = y_curve(mid).cpu().numpy(), y_curve(fin).cpu().numpy()
        row = {"id": ib, "tok": w, "D": float((ea - eb).norm())}
        for name, y in (("mid", ym), ("final", yf)):
            t1, t9, W, n1, n9, bs = width(TS, y)
            row.update({f"t10_{name}": t1, f"t90_{name}": t9, f"W_{name}": W,
                        f"ncross10_{name}": n1, f"ncross90_{name}": n9, f"backstep_{name}": bs,
                        f"y0_{name}": float(y[0]), f"y1_{name}": float(y[-1])})
        rows.append(row)
        curves[w] = (ym, yf)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
    # endpoint consistency: interpolated endpoint B equals a direct forward of the token-B prompt
    ids = torch.tensor([r.ids[:-1] + [ids_b[0]]], device=dev)
    with torch.no_grad():
        m.transformer(input_ids=ids, use_cache=False)
    direct = r.cap["final"][0]
    _, fin = r.run(r.wte[ids_b[0]].unsqueeze(0))
    print("direct vs inputs_embeds endpoint max abs diff:", float((direct - fin[0]).abs().max()))
    json.dump(rows, open(os.path.join(RESULTS, "s1_sanity.json"), "w"), indent=1)

    fig, axes = plt.subplots(4, 5, figsize=(16, 12), sharex=True, sharey=True)
    for ax, row in zip(axes.flat, rows):
        ym, yf = curves[row["tok"]]
        ax.plot(TS, ym, color=CVD[0], ls="-", lw=1.5, label="block 18")
        ax.plot(TS, yf, color=CVD[1], ls="--", lw=1.5, label="block 35")
        for name, c, mk in (("mid", CVD[0], "o"), ("final", CVD[1], "s")):
            ax.plot([row[f"t10_{name}"], row[f"t90_{name}"]], [0.1, 0.9], mk, color=c, ms=5)
        ax.axhline(0.1, color="gray", lw=0.5, ls=":")
        ax.axhline(0.9, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{row['tok']!r}  D={row['D']:.2f}\nW18={row['W_mid']:.3f}  W35={row['W_final']:.3f}",
                     fontsize=9)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("y(t)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "s1_sanity_curves.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
