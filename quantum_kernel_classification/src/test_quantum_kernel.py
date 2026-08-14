"""
Sanity checks for quantum_kernel.py:
  1. Brute-force cross-check: rebuild the ZZ feature-map circuit with explicit
     dense matrices (kron'd Hadamards + explicitly-looped diagonal phase) for
     small n, and compare against the vectorized simulator bit-for-bit.
  2. Physical sanity: statevectors have unit norm (circuit is unitary).
  3. Kernel sanity: K is symmetric, diagonal ~1, and PSD (up to float noise).
"""
import numpy as np
from quantum_kernel import (
    feature_map_statevectors, quantum_kernel_matrix, _pm1_matrix,
)

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2.0)


def brute_force_statevector(x, n_qubits, reps):
    """Explicit dense-matrix construction, independent implementation."""
    Hn = H
    for _ in range(n_qubits - 1):
        Hn = np.kron(Hn, H)

    dim = 2 ** n_qubits
    theta = np.zeros(dim)
    for z in range(dim):
        bits = [(z >> i) & 1 for i in range(n_qubits)]
        s = [1 - 2 * b for b in bits]  # (-1)^{z_i}
        val = sum(x[i] * s[i] for i in range(n_qubits))
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                val += (np.pi - x[i]) * (np.pi - x[j]) * s[i] * s[j]
        theta[z] = val
    D = np.diag(np.exp(1j * theta))

    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    for _ in range(reps):
        state = Hn @ state
        state = D @ state
    return state


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    for n_qubits in (1, 2, 3, 4):
        x = rng.uniform(0, 2 * np.pi, size=n_qubits)
        for reps in (1, 2, 3):
            fast = feature_map_statevectors(x[None, :], n_qubits, reps=reps)[0]
            slow = brute_force_statevector(x, n_qubits, reps)
            err = np.max(np.abs(fast - slow))
            assert err < 1e-10, f"mismatch n={n_qubits} reps={reps}: err={err}"
    print("PASS: vectorized simulator matches brute-force dense-matrix construction")


def test_unitary_norm():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 2 * np.pi, size=(50, 5))
    psi = feature_map_statevectors(X, n_qubits=5, reps=2)
    norms = np.sum(np.abs(psi) ** 2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9), norms
    print("PASS: all statevectors have unit norm (circuit is unitary)")


def test_kernel_properties():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 2 * np.pi, size=(30, 4))
    K = quantum_kernel_matrix(X, X, n_qubits=4, reps=2)
    assert np.allclose(K, K.T, atol=1e-9), "kernel not symmetric"
    assert np.allclose(np.diag(K), 1.0, atol=1e-9), "self-kernel entries not 1"
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-8, f"kernel not PSD, min eigenvalue={eigvals.min()}"
    print("PASS: kernel matrix is symmetric, PSD, with unit diagonal")


def test_pm1_matrix():
    S = _pm1_matrix(3)
    assert S.shape == (8, 3)
    assert set(np.unique(S)) == {-1, 1}
    print("PASS: +-1 basis matrix has correct shape/values")


if __name__ == "__main__":
    test_pm1_matrix()
    test_matches_brute_force()
    test_unitary_norm()
    test_kernel_properties()
    print("\nAll sanity checks passed.")
