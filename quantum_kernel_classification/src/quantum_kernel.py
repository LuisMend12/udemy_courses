"""
From-scratch exact simulation of the Havlicek et al. (2019) ZZ feature-map
quantum kernel, implemented without any quantum-computing library.

Circuit (per repetition):
    |0>^n --H^{⊗n}--> uniform superposition
                    --diag(exp(i*theta(x)))--> phase-encoded state

theta(z, x) = sum_i x_i * (-1)^{z_i}
            + sum_{i<j} (pi - x_i)(pi - x_j) * (-1)^{z_i} * (-1)^{z_j}

which is the "full entanglement" Pauli-ZZ feature map. Repeated `reps` times.

Because the diagonal unitary is diagonal in the computational basis, the
whole circuit reduces to alternating a batched Walsh-Hadamard transform with
an elementwise complex phase multiply -- no gate-by-gate simulation or
quantum library is needed, and it is exact (not a Monte-Carlo estimate).

The fidelity kernel is k(x, x') = |<phi(x)|phi(x')>|^2.
"""
import numpy as np


def _pm1_matrix(n_qubits: int) -> np.ndarray:
    """Rows = all length-n_qubits +-1 vectors, i.e. (-1)^{z_i} for z in {0,..,2^n-1}."""
    idx = np.arange(2 ** n_qubits)
    bits = ((idx[:, None] >> np.arange(n_qubits)[None, :]) & 1)
    return 1 - 2 * bits  # shape (2**n_qubits, n_qubits), entries in {+1,-1}


def _hadamard_transform_batch(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply H^{⊗n_qubits} to each row of `state` (shape (N, 2**n_qubits))."""
    n_samples = state.shape[0]
    s = state.reshape((n_samples,) + (2,) * n_qubits)
    for axis in range(1, n_qubits + 1):
        s = np.moveaxis(s, axis, -1)
        shape = s.shape
        s = s.reshape(-1, 2)
        a, b = s[:, 0], s[:, 1]
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        s = np.stack([(a + b) * inv_sqrt2, (a - b) * inv_sqrt2], axis=-1).reshape(shape)
        s = np.moveaxis(s, -1, axis)
    return s.reshape(n_samples, 2 ** n_qubits)


def _phase_angles(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """theta(z, x) for every sample in X (N, n_qubits) and every basis state z.

    Uses sum_{i<j} S_i S_j u_i u_j = 0.5*[(S.u)^2 - sum_i u_i^2] with u = pi - x,
    which follows from S_i^2 = 1, to avoid an O(n^2) loop over qubit pairs.
    """
    linear = X @ S.T                                   # (N, 2**n)
    u = np.pi - X
    su = u @ S.T                                        # (N, 2**n)
    quadratic = 0.5 * (su ** 2 - np.sum(u ** 2, axis=1, keepdims=True))
    return linear + quadratic


def feature_map_statevectors(X: np.ndarray, n_qubits: int, reps: int = 2) -> np.ndarray:
    """Exact statevectors |phi(x)> for each row of X. Returns (N, 2**n_qubits) complex128."""
    X = np.asarray(X, dtype=float)
    assert X.shape[1] == n_qubits, f"expected {n_qubits} features, got {X.shape[1]}"
    S = _pm1_matrix(n_qubits)
    n_samples = X.shape[0]
    state = np.zeros((n_samples, 2 ** n_qubits), dtype=complex)
    state[:, 0] = 1.0  # |0...0>
    for _ in range(reps):
        state = _hadamard_transform_batch(state, n_qubits)
        theta = _phase_angles(X, S)
        state = state * np.exp(1j * theta)
    return state


def quantum_kernel_matrix(X1: np.ndarray, X2: np.ndarray, n_qubits: int, reps: int = 2) -> np.ndarray:
    """Fidelity kernel k(x,x') = |<phi(x)|phi(x')>|^2 between rows of X1 and X2."""
    psi1 = feature_map_statevectors(X1, n_qubits, reps)
    psi2 = psi1 if X2 is X1 else feature_map_statevectors(X2, n_qubits, reps)
    gram = psi1 @ psi2.conj().T
    return np.abs(gram) ** 2


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Cristianini et al. (2001) kernel-target alignment, y in {-1, +1}."""
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    yy = y @ y.T
    num = np.sum(K * yy)
    den = np.sqrt(np.sum(K * K) * np.sum(yy * yy))
    return float(num / den)


def random_haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR decomposition of a complex Gaussian matrix."""
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph  # correct Haar measure (Mezzadri, 2006)
