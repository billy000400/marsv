"""S0: one-layer hook/transcoder validation on a small mixed batch of prompts."""
import json
import os
import sys

import torch

from common import OUT, MLPTap, encode, load_model, load_transcoder, setup_torch

PROMPTS = ["The capital of Japan is", "Sushi originates from the country of",
           "The headquarters of SoundCloud are in the city of", "People in Germany speak",
           "The quick brown fox jumps over the lazy dog.", "def add(a, b):\n    return a + b",
           "In 1969, astronauts first walked on the", "She opened the door and saw that the room was"]


def main(layer):
    setup_torch()
    tok, model = load_model()
    tap = MLPTap(model)
    xs, ys, resid_ok = [], [], []
    for p in PROMPTS:
        enc = tok(p, return_tensors="pt").to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        block = model.model.layers[layer]
        # MLP input must equal the post-attention RMSNorm of the mid-block residual; check via the block identity
        # resid_out = resid_mid + mlp(norm(resid_mid)).
        resid_mid = hs[layer + 1] - tap.y[layer]
        resid_ok.append(torch.allclose(block.post_attention_layernorm(resid_mid), tap.x[layer], atol=1e-3, rtol=1e-3))
        xs.append(tap.x[layer][0])
        ys.append(tap.y[layer][0])
    first = torch.cat([torch.arange(len(x)) == 0 for x in xs]).to(xs[0].device)
    x, y = torch.cat(xs), torch.cat(ys)
    W_enc, b_enc, W_dec, b_dec = load_transcoder(layer)
    a = encode(x, W_enc, b_enc)
    y_hat = a @ W_dec + b_dec
    rel = ((y_hat - y).norm(dim=-1) / y.norm(dim=-1))
    res = dict(layer=layer, n_prompts=len(PROMPTS), n_tokens=len(x), d_in=x.shape[-1], d_out=y.shape[-1],
               W_enc_shape=list(W_enc.shape), W_dec_shape=list(W_dec.shape), finite=bool(torch.isfinite(a).all()),
               mlp_input_is_post_attention_norm=all(resid_ok), recon_shape_matches=y_hat.shape == y.shape,
               mean_active_features=a.gt(0).sum(-1).float().mean().item(),
               mean_active_features_excl_first_token=a[~first].gt(0).sum(-1).float().mean().item(),
               rel_l2_error_mean=rel.mean().item(), rel_l2_error_excl_first_token=rel[~first].mean().item(),
               rel_l2_error_first_token=rel[first].mean().item(),
               rel_l2_error_last_token=torch.stack([rel[i] for i in (torch.cumsum(torch.tensor([len(v) for v in xs]), 0) - 1)]).mean().item())
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, f"s0_validation_layer{layer}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
