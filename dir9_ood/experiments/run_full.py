"""Full Direction-#9 sweep: plateau variants vs baselines across measurement points & OOD sets.

Writes results/auroc_table.csv with one row per (ood_set, method, measurement_point, auroc).
CPU-only (V100 has no kernels for this torch). Economized for the shared box:
  seq_len=32, n_dirs=8, N per set ~40 by default (override via argv: N seq_len n_dirs).
"""
import os, sys, time, csv
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", "/mars-vol/.cache/huggingface")
import numpy as np
import plateau_score as ps

t0 = time.time()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NDIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
EPS = 6.0
POINTS = ["input", "resid3", "resid6", "resid9"]
OOD_SETS = ["random", "shuffled", "code"]
HERE = os.path.dirname(__file__)

texts = [l.rstrip("\n") for l in open(os.path.join(HERE, "..", "data", "fineweb_sample.txt"))]
all_ids = ps.encode_fixed(texts, seq_len=SEQ)
# ID split: first N for the test set, a separate N for fitting Mahalanobis
id_test = all_ids[:N]
id_fit = all_ids[N : 2 * N]
ood = {k: ps.make_ood(id_test, k, seed=0) for k in OOD_SETS}
print(f"N={N} seq={SEQ} ndirs={NDIRS} | id_test {id_test.shape} id_fit {id_fit.shape}", flush=True)


def score_set(ids):
    """All methods for one set of sequences. Returns nested dict method->point->array (or point None)."""
    out = {}
    acts, msp = ps.collect_all(ids, POINTS, seq_len=SEQ)
    out["acts"] = acts
    out["baseline-MSP"] = {None: msp}
    out["baseline-L2norm"] = {p: np.linalg.norm(acts[p], axis=1) for p in POINTS}
    jac = ps.jacobian_all(ids, POINTS, seq_len=SEQ)
    out["plateau-jacobian"] = jac
    pert = {}
    for p in POINTS:
        pert[p] = ps.perturbation_score(ids, p, n_dirs=NDIRS, eps=EPS, seq_len=SEQ)
        print(f"    perturb {p} done {time.time()-t0:.0f}s", flush=True)
    out["plateau-perturbation"] = pert
    return out


print("scoring id_fit (for mahalanobis)...", flush=True)
fit_acts, _ = ps.collect_all(id_fit, POINTS, seq_len=SEQ)
print(f"scoring id_test... {time.time()-t0:.0f}s", flush=True)
S_id = score_set(id_test)
S_ood = {}
for k in OOD_SETS:
    print(f"scoring ood={k}... {time.time()-t0:.0f}s", flush=True)
    S_ood[k] = score_set(ood[k])

# Mahalanobis per point (fit on id_fit, distance for id_test & each ood)
maha = {}
for p in POINTS:
    maha[p] = {"id": ps.mahalanobis(fit_acts[p], S_id["acts"][p])}
    for k in OOD_SETS:
        maha[p][k] = ps.mahalanobis(fit_acts[p], S_ood[k]["acts"][p])

# ---- assemble AUROC table
rows = []
METHODS_POINTED = ["plateau-perturbation", "plateau-jacobian", "baseline-L2norm"]
for k in OOD_SETS:
    # MSP (no point)
    sid = S_id["baseline-MSP"][None]
    sood = S_ood[k]["baseline-MSP"][None]
    a = ps.auroc(np.concatenate([sid, sood]), np.concatenate([np.zeros(len(sid)), np.ones(len(sood))]))
    rows.append(["fineweb-vs", k, "baseline-MSP", "n/a", round(float(a), 4)])
    # Mahalanobis (per point)
    for p in POINTS:
        sid = maha[p]["id"]; sood = maha[p][k]
        a = ps.auroc(np.concatenate([sid, sood]), np.concatenate([np.zeros(len(sid)), np.ones(len(sood))]))
        rows.append(["fineweb-vs", k, "baseline-mahalanobis", p, round(float(a), 4)])
    # pointed methods
    for m in METHODS_POINTED:
        for p in POINTS:
            sid = S_id[m][p]; sood = S_ood[k][m][p]
            a = ps.auroc(np.concatenate([sid, sood]), np.concatenate([np.zeros(len(sid)), np.ones(len(sood))]))
            rows.append(["fineweb-vs", k, m, p, round(float(a), 4)])

os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
csv_path = os.path.join(HERE, "..", "results", "auroc_table.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "ood_set", "method", "measurement_point", "auroc"])
    w.writerows(rows)
# dump ALL raw scores keyed "{set}|{method}|{point}" for plotting (incl. OOD sets + mahalanobis)
dump = {}
def add(setname, S):
    for m in ["plateau-perturbation", "plateau-jacobian", "baseline-L2norm"]:
        for p in POINTS:
            dump[f"{setname}|{m}|{p}"] = S[m][p]
    dump[f"{setname}|baseline-MSP|n/a"] = S["baseline-MSP"][None]
add("id", S_id)
for k in OOD_SETS:
    add(k, S_ood[k])
for p in POINTS:
    dump[f"id|baseline-mahalanobis|{p}"] = maha[p]["id"]
    for k in OOD_SETS:
        dump[f"{k}|baseline-mahalanobis|{p}"] = maha[p][k]
np.savez(os.path.join(HERE, "..", "results", "scores_full.npz"), **dump)
print(f"\nwrote {csv_path} ({len(rows)} rows) in {time.time()-t0:.0f}s")
for r in rows:
    print(r)
