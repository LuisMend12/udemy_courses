"""Classical kernel functions with the same (X1, X2) -> Gram matrix signature
as quantum_kernel.quantum_kernel_matrix, so all kernels can be driven through
one generic evaluation pipeline."""
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
