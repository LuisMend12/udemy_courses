"""
Synthetic "quantum-advantage" classification dataset, following the
construction of Havlicek et al., "Supervised learning with quantum-enhanced
feature spaces", Nature 567 (2019).

Idea: label points by the sign of the expectation value of a fixed observable
measured in a randomly-rotated basis, where the state being measured is the
ZZ feature map applied to x:

    M = V P V^dagger,   P = diag(f),  f_z = (-1)^{parity(z & mask)}
    label(x) = sign( <phi(x)| M |phi(x)> )
    margin(x) = | <phi(x)| M |phi(x)> |

V is a fixed Haar-random unitary and `mask` picks a fixed non-trivial
Pauli-Z-string observable, both frozen by a random seed. Points are drawn
uniformly from [0, 2*pi)^n_qubits and only those with margin >= gamma are
kept, which by construction makes the resulting dataset perfectly separable
by the quantum kernel with margin >= gamma (a large-margin positive control).
There is no guarantee an easy classical decision boundary exists for it.
"""
import numpy as np
from quantum_kernel import feature_map_statevectors, random_haar_unitary


def _fixed_parity_observable(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    dim = 2 ** n_qubits
    mask = int(rng.integers(1, dim))  # nonzero -> non-trivial observable
    idx = np.arange(dim)
    parity = np.zeros(dim, dtype=int)
    m = mask
    bit = 0
    while m:
        if m & 1:
            parity ^= (idx >> bit) & 1
        m >>= 1
        bit += 1
    return 1 - 2 * parity  # (-1)^parity, shape (dim,)


def generate_quantum_advantage_dataset(
    n_qubits: int,
    n_samples: int,
    gamma: float = 0.3,
    reps: int = 2,
    seed: int = 0,
    pool_batch: int = 4000,
    max_pool_batches: int = 200,
):
    """Returns (X, y, info) with X in [0, 2*pi)^n_qubits, y in {-1, +1}, class-balanced."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    V = random_haar_unitary(dim, rng)
    f = _fixed_parity_observable(n_qubits, rng)

    per_class = n_samples // 2
    kept_X, kept_y = [], []
    n_pos = n_neg = 0

    for _ in range(max_pool_batches):
        if n_pos >= per_class and n_neg >= per_class:
            break
        pool = rng.uniform(0, 2 * np.pi, size=(pool_batch, n_qubits))
        psi = feature_map_statevectors(pool, n_qubits, reps=reps)
        chi = psi @ V.conj()
        probs = np.abs(chi) ** 2
        expectation = probs @ f
        margin = np.abs(expectation)
        label = np.sign(expectation)

        ok = margin >= gamma
        for x, y in zip(pool[ok], label[ok]):
            if y > 0 and n_pos < per_class:
                kept_X.append(x); kept_y.append(1.0); n_pos += 1
            elif y < 0 and n_neg < per_class:
                kept_X.append(x); kept_y.append(-1.0); n_neg += 1

    X = np.array(kept_X)
    y = np.array(kept_y)
    if len(y) < 2 * per_class:
        raise RuntimeError(
            f"only found {n_pos} positive / {n_neg} negative samples with margin>={gamma} "
            f"after {max_pool_batches} pool batches; lower gamma or increase pool_batch."
        )
    info = {"V": V, "f": f, "n_qubits": n_qubits, "gamma": gamma, "reps": reps}
    return X, y, info


if __name__ == "__main__":
    X, y, info = generate_quantum_advantage_dataset(n_qubits=4, n_samples=200, gamma=0.3, seed=0)
    print("X shape:", X.shape, "class balance:", np.mean(y == 1))
