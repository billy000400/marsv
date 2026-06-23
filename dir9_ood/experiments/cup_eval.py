"""Evaluate OOD detection with the REAL cupbearer package (operator feedback 2026-06-21).

Runs in the ISOLATED conda env /mars-vol/marsv/dir9_ood/cupenv (its own numpy<2 + torch +
the cupbearer GitHub build installed editable). Loads the GPT-2 last-token activations that
extract_acts.py produced in the base env, wraps them in a cupbearer FeatureExtractor that
returns precomputed features, and runs cupbearer's actual statistical detectors:
  - MahalanobisDetector(relative=False)  -> cup-maha   (naive Mahalanobis)
  - MahalanobisDetector(relative=True)   -> cup-RMD    (relative Mahalanobis)
  - QuantumEntropyDetector               -> cup-QUE    (SPECTRE-style quantum entropy)
  - SpectralSignatureDetector            -> cup-spectral
Each is fit on ID FineWeb (idtrain, 1000 seqs) and scores ID-test vs each OOD set.
Output: results/auroc_cupbearer.csv  [ood_set, method, measurement_point, auroc].

This is the genuine package end-to-end (not vendored math), in a separate env so the shared
base env's torch 2.9.0+cu130 / numpy 2.3.3 are untouched.
"""
import os, sys, csv
import numpy as np
import torch

torch.set_num_threads(2)
ACTS = os.path.join(os.path.dirname(__file__), "..", "results", "acts")
OUTCSV = os.path.join(os.path.dirname(__file__), "..", "results", "auroc_cupbearer.csv")
POINTS = ["input", "resid3", "resid6", "resid9"]
OOD_SETS = ["random", "shuffled", "code"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from cupbearer.detectors import (
    MahalanobisDetector, QuantumEntropyDetector, SpectralSignatureDetector,
)
from cupbearer.detectors.extractors.core import FeatureExtractor


def auroc(scores, labels):
    """AUROC, label==1 positive (OOD). Pure numpy Mann-Whitney with tie handling."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores) + 1)
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    for i in range(len(uniq)):
        lo = csum[i] - counts[i]
        avg[i] = (lo + 1 + csum[i]) / 2.0
    ranks = np.array([avg[i] for i in inv])
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class PrecomputedExtractor(FeatureExtractor):
    """Returns a single precomputed activation tensor keyed by `point`. The dataset yields
    integer row-indices into the current split's array (set via set_array)."""
    def __init__(self, point):
        super().__init__(feature_names=[point])
        self.point = point
        self._arr = None

    def set_array(self, arr_np):
        self._arr = torch.from_numpy(np.ascontiguousarray(arr_np)).float().to(DEVICE)

    def compute_features(self, inputs):
        idx = inputs.long().to(self._arr.device)
        return {self.point: self._arr[idx]}

    def __call__(self, inputs):            # skip the model/device plumbing in the base class
        return self.compute_features(inputs)


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i, 0   # (input=index, dummy label)


def load(split, point):
    return np.load(os.path.join(ACTS, f"{split}__{point}.npy"))


def run_detector(make_detector, point, needs_untrusted, train_kwargs):
    ext = PrecomputedExtractor(point)
    det = make_detector(ext)
    idtrain = load("idtrain", point)
    ext.set_array(idtrain)
    n = idtrain.shape[0]
    tds = IndexDataset(n)
    kw = dict(trusted_data=tds, batch_size=128, shuffle=False, num_workers=0)
    if needs_untrusted:
        kw["untrusted_data"] = tds         # ID stats as the untrusted reference (unsupervised)
    det.train(model=None, **kw, **train_kwargs)

    def score(split):
        arr = load(split, point)
        ext.set_array(arr)
        with torch.no_grad():
            s = det.compute_scores(torch.arange(arr.shape[0]))
        return s.detach().cpu().numpy()

    id_s = score("idtest")
    out = {}
    for ood in OOD_SETS:
        ood_s = score(ood)
        sc = np.concatenate([id_s, ood_s])
        lb = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
        out[ood] = auroc(sc, lb)
    return out


# `relative` is a kwarg of MahalanobisDetector.post_covariance_training, reached via train(**kwargs)
# -> _train -> _finalize_training -> post_covariance_training. So it belongs in train_kwargs.
METHODS = [
    ("cup-maha",     lambda e: MahalanobisDetector(feature_extractor=e),       False, {"relative": False}),
    ("cup-RMD",      lambda e: MahalanobisDetector(feature_extractor=e),       False, {"relative": True}),
    ("cup-QUE",      lambda e: QuantumEntropyDetector(feature_extractor=e),    True,  {}),
    ("cup-spectral", lambda e: SpectralSignatureDetector(feature_extractor=e), True,  {}),
]


def main():
    import cupbearer
    print("cupbearer from", cupbearer.__file__, "| device", DEVICE, flush=True)
    rows = [("ood_set", "method", "measurement_point", "auroc")]
    for name, mk, needs_unt, tkw in METHODS:
        for point in POINTS:
            try:
                res = run_detector(mk, point, needs_unt, tkw)
            except Exception as e:
                print(f"  {name}@{point} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            for ood in OOD_SETS:
                rows.append((ood, name, point, f"{res[ood]:.4f}"))
            print(f"  {name}@{point}: " + " ".join(f"{o}={res[o]:.3f}" for o in OOD_SETS), flush=True)
    with open(OUTCSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("WROTE", OUTCSV, f"({len(rows)-1} rows)", flush=True)


if __name__ == "__main__":
    main()
