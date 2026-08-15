"""Classical kernel functions with the same (X1, X2) -> Gram matrix signature
as quantum_kernel.quantum_kernel_matrix, so all kernels can be driven through
one generic evaluation pipeline."""
import numpy as np
from functools import partial
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


def classical_kernel_fn(kind: str, **params):
    if kind == "rbf":
        return partial(rbf_kernel, gamma=params["gamma"])
    if kind == "poly":
        return partial(polynomial_kernel, degree=params["degree"], coef0=1, gamma=params.get("gamma", 1.0))
    if kind == "linear":
        return linear_kernel
    raise ValueError(kind)


def normalize_kernel(K: np.ndarray) -> np.ndarray:
    """Rescale to unit diagonal: Knorm(x,x') = K(x,x') / sqrt(K(x,x) K(x',x')).

    The quantum fidelity kernel already has unit diagonal by construction
    (|<phi(x)|phi(x)>|^2 = 1), so this is a no-op there; for classical
    kernels (especially poly at degree>=3 on features scaled to [0, 2*pi))
    it keeps entries in [-1, 1], which both makes kernel-target alignment
    comparable across kernel families and keeps libsvm's QP well-conditioned
    (unnormalized poly entries can reach 1e4-1e5 and stall the solver).
    """
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / np.outer(d, d)
