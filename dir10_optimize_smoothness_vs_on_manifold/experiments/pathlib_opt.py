"""Core machinery for the combined-objective lambda sweep (S3).

Pieces:
  TailRunner   — GPU-resident layers 28..31 + final norm + lm_head. Given a base
                 prompt's cached layer-28 hidden states (all positions) and a batch
                 of replacement last-token activations, returns the 8-bin weekday
                 behavior distribution. Differentiable w.r.t. the injected vector.
  spline_matrix — fixed linear map M [N_wp, n_ctrl] from control points to
                 evaluation waypoints for a NATURAL cubic spline (path param).
  kinetic_energy — discrete Dirichlet / kinetic energy of a sampled curve.

Injection convention (Appendix-style pullback): the path lives in the first-32 PCA
components of the layer-28 activation space. For each base prompt we OVERWRITE its
own first-32 PCA coordinates (at the last token) with the path waypoint, keeping the
higher PCA components and the orthogonal residual fixed. Equivalent shift:
    injected = a_p + V32^T @ (w - V32 @ (a_p - mean))
"""
import os
import numpy as np
import torch

import common as C


# ---------- spline path parameterization ----------
def spline_matrix(n_ctrl=10, n_wp=20):
    """Natural-cubic-spline linear map: W[n_wp, d] = M[n_wp, n_ctrl] @ Ctrl[n_ctrl, d].
    Control knots and evaluation waypoints are uniform on [0, 1]."""
    from scipy.interpolate import CubicSpline
    tk = np.linspace(0, 1, n_ctrl)
    tq = np.linspace(0, 1, n_wp)
    M = np.zeros((n_wp, n_ctrl))
    for i in range(n_ctrl):
        e = np.zeros(n_ctrl); e[i] = 1.0
        cs = CubicSpline(tk, e, bc_type="natural")
        M[:, i] = cs(tq)
    return M


def kinetic_energy(W):
    """Discrete kinetic energy of curve W [n_wp, d]:  sum_i |Δx_i|^2 / Δt,
    Δt = 1/(n_wp-1). This is the finite-difference approx of ∫|x'(t)|^2 dt."""
    n = W.shape[0]
    dt = 1.0 / (n - 1)
    diffs = W[1:] - W[:-1]
    return (diffs.pow(2).sum()) / dt


# ---------- model tail ----------
class TailRunner:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(C.MODEL_ID)
        # Load fully on CPU (real tensors, ~16 GB in the box's ~230 GB free RAM).
        # base_hidden runs on CPU; the tail is then moved to GPU for the sweep.
        model = AutoModelForCausalLM.from_pretrained(
            C.MODEL_ID, dtype=torch.bfloat16, device_map={"": "cpu"}).eval()
        self.model = model
        self.cfg = model.config
        self.n_start = C.LAYER  # inject at output of block LAYER == input of block index LAYER
        self.wids = C.weekday_token_ids(self.tok)
        self.on_gpu = False

    @torch.no_grad()
    def base_hidden(self, prompt):
        """Full layer-28 hidden states [seq, hidden] for a prompt (CPU forward)."""
        enc = self.tok(prompt, return_tensors="pt")
        out = self.model(enc.input_ids, output_hidden_states=True)
        return out.hidden_states[C.LAYER][0].to("cuda").float()  # [seq, hidden]

    def move_tail_to_gpu(self):
        """Move the tail (blocks LAYER.., norm, rotary, lm_head) to GPU, drop the
        lower blocks + embed to free RAM. Call AFTER caching base hidden states."""
        lm = self.model.model
        self.layers = [lm.layers[i].to("cuda") for i in
                       range(self.n_start, self.cfg.num_hidden_layers)]
        self.norm = lm.norm.to("cuda")
        self.lm_head = self.model.lm_head.to("cuda")
        self.rotary = lm.rotary_emb.to("cuda")
        for i in range(self.n_start):
            lm.layers[i] = None
        lm.embed_tokens = None
        self.on_gpu = True
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def behavior(self, base_hs, injected_last):
        """base_hs [seq, hidden] (GPU, bf16-castable); injected_last [B, hidden].
        Returns dist8 [B, 8] (float32, differentiable)."""
        seq = base_hs.shape[0]
        B = injected_last.shape[0]
        hs = base_hs.unsqueeze(0).expand(B, seq, -1).to(torch.bfloat16).clone()
        hs[:, -1, :] = injected_last.to(torch.bfloat16)
        pos = torch.arange(seq, device="cuda").unsqueeze(0)
        cos, sin = self.rotary(hs, pos)
        cmask = torch.full((seq, seq), float("-inf"), device="cuda", dtype=hs.dtype)
        cmask = torch.triu(cmask, diagonal=1).view(1, 1, seq, seq)
        for layer in self.layers:
            hs = layer(hs, attention_mask=cmask, position_ids=pos,
                       position_embeddings=(cos, sin))
        last = self.norm(hs[:, -1, :])
        logits = self.lm_head(last).float()               # [B, vocab]
        probs = torch.softmax(logits, dim=-1)
        cols = []
        for w in C.WEEKDAYS:
            cols.append(probs[:, self.wids[w]].sum(dim=1))
        wd = torch.stack(cols, dim=1)                     # [B, 7]
        other = (1.0 - wd.sum(dim=1, keepdim=True)).clamp_min(0.0)
        return torch.cat([wd, other], dim=1)              # [B, 8]
