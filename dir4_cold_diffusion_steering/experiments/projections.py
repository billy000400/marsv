"""Projection utilities + the analytic projection-preserving corrector for ColdSteer.

The ColdSteer parameterization is  ĥ = z + P_{v⊥} r,  where z = h + α v is the raw
steered activation and r is a correction constrained to the subspace orthogonal to v.
By construction this preserves the steering projection along v exactly:

    ⟨ĥ - h, v̂⟩ = ⟨z - h, v̂⟩ = α |v|.

`cov_aligned_shift` gives the ANALYTIC optimal r for a Gaussian activation model:
among all constant shifts Δ that achieve the target v-projection α|v|, it returns the
one of minimum whitened norm  Δ^T Σ^{-1} Δ  (i.e. the smallest on-manifold movement).
That minimizer is  Δ = Σ v̂ · α|v| / (v̂^T Σ v̂),  which is a rotation of the raw shift
α v toward how activations actually covary. Kantorovich's inequality guarantees its
Mahalanobis penalty is ≤ that of the raw shift α v, with equality only when v is an
eigenvector of Σ — so it provably reduces off-manifold distance while matching projection.
"""
import numpy as np

EPS = 1e-8


def normalize_vector(v, eps=EPS):
    """Unit-normalize along the last axis (works batched or unbatched)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def apply_steering(h, v, alpha):
    """Raw linear steering z = h + alpha * v (broadcasts over batch)."""
    return h + alpha * v


def projection_along(x, v):
    """Scalar projection coefficient <x, v̂> onto the unit direction of v."""
    vhat = normalize_vector(v)
    return x @ vhat


def project_orthogonal(x, v):
    """Component of x orthogonal to v:  x - <x, v̂> v̂."""
    vhat = normalize_vector(v)
    coeff = x @ vhat
    if x.ndim > 1:
        return x - np.outer(coeff, vhat)
    return x - coeff * vhat


def retain_projection_update(z, residual, v):
    """ĥ = z + P_{v⊥}(residual). Preserves ⟨ĥ - h, v⟩ = ⟨z - h, v⟩ for any residual."""
    return z + project_orthogonal(residual, v)


def cov_aligned_shift(v, cov, alpha):
    """Analytic optimal ColdSteer shift: the min-whitened-norm constant shift Δ with
    ⟨Δ, v̂⟩ = alpha·|v|, given activation covariance `cov`. Δ = Σv̂ · α|v| / (v̂^T Σ v̂)."""
    vn = np.linalg.norm(v)
    vhat = v / (vn + EPS)
    Sv = cov @ vhat
    denom = float(vhat @ Sv)
    beta = alpha * vn / denom
    return beta * Sv


# ---------------------------------------------------------------------------
# Unit tests (run:  python projections.py)
# ---------------------------------------------------------------------------
def _tests():
    rng = np.random.RandomState(0)
    d = 32
    v = rng.randn(d).astype(np.float64)
    h = rng.randn(8, d).astype(np.float64)

    # 1. alpha=0 returns the original activation.
    assert np.allclose(apply_steering(h, v, 0.0), h), "alpha=0 must be identity"

    # 2. Orthogonal projection is (numerically) orthogonal to v.
    o = project_orthogonal(h, v)
    assert np.allclose(o @ (v / np.linalg.norm(v)), 0.0, atol=1e-8), "P_perp not orthogonal"

    # 3. retain_projection_update keeps ⟨ĥ - h, v⟩ = ⟨z - h, v⟩ for arbitrary residual.
    alpha = 3.0
    z = apply_steering(h, v, alpha)
    residual = rng.randn(8, d)
    hhat = retain_projection_update(z, residual, v)
    lhs = (hhat - h) @ v
    rhs = (z - h) @ v
    assert np.allclose(lhs, rhs, atol=1e-8), "projection along v not preserved"
    # and equals the intended alpha*|v|^2  (since v not unit here) -> alpha * v·v
    assert np.allclose(rhs, alpha * (v @ v)), "raw projection mismatch"

    # 4. normalize_vector stable batched and unbatched.
    assert np.allclose(np.linalg.norm(normalize_vector(v)), 1.0)
    vb = rng.randn(5, d)
    assert np.allclose(np.linalg.norm(normalize_vector(vb), axis=-1), 1.0)

    # 5. cov_aligned_shift: matches the target v-projection AND has <= Mahalanobis
    #    penalty than the raw shift alpha*v (Kantorovich), strictly < for generic v.
    A = rng.randn(d, d)
    cov = A @ A.T + np.eye(d)          # SPD covariance
    prec = np.linalg.inv(cov)
    vn = np.linalg.norm(v)
    delta = cov_aligned_shift(v, cov, alpha)
    # matched projection: <Δ, v̂> == alpha*|v|
    assert np.allclose((delta @ v) / vn, alpha * vn, atol=1e-6), "cov shift projection off"
    # fits the ĥ = z + P_{v⊥} r form: (Δ - alpha*v) ⟂ v
    r = delta - alpha * v
    assert np.allclose((r @ v) / vn, 0.0, atol=1e-6), "cov shift not in z+P_perp form"
    # Mahalanobis penalty strictly reduced vs raw
    pen_raw = alpha**2 * (v @ prec @ v)
    pen_corr = delta @ prec @ delta
    assert pen_corr < pen_raw + 1e-9, "cov shift did not reduce Mahalanobis penalty"
    assert pen_corr < 0.999 * pen_raw, "cov shift should strictly reduce for generic v"

    print("projections.py: all unit tests PASSED")
    print(f"  raw Mahalanobis penalty  = {pen_raw:.4f}")
    print(f"  cov-aligned penalty      = {pen_corr:.4f}  ({pen_corr/pen_raw:.2%} of raw)")


if __name__ == "__main__":
    _tests()
