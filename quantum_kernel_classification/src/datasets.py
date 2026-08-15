"""Real-dataset loader: MiniBooNE particle identification (UCI / OpenML), a
real high-energy-physics signal-vs-background classification task -- Fermilab
MiniBooNE electron-neutrino appearance search, classifying electron-neutrino
signal events against muon-neutrino background events from 50 reconstructed
kinematic variables per event. PCA-reduce to n_qubits dims, scale to
[0, 2*pi) for the quantum feature map (classical kernels are fed the same
representation, so the comparison isolates the kernel choice rather than
confounding it with different preprocessing)."""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

N_SUBSAMPLE = 400


def _load_miniboone_raw():
    d = fetch_openml("MiniBooNE", version=1, as_frame=True, parser="auto")
    X = d.data.to_numpy(dtype=float)
    y_raw = d.target.to_numpy()
    y = np.where(y_raw == "True", 1.0, -1.0)  # electron-neutrino signal vs muon-neutrino background

    # ~0.36% of rows use -999.0 as a missing-value sentinel in several
    # columns; left in, they dominate StandardScaler/PCA, so we drop them.
    clean = ~np.any(X == -999.0, axis=1)
    return X[clean], y[clean]


def load_pca_reduced(name: str, n_qubits: int, seed: int = 0):
    if name != "miniboone":
        raise ValueError(name)

    X_full, y_full = _load_miniboone_raw()
    # Fix the subsample (independent of n_qubits) so different qubit counts
    # compare the same underlying points at different PCA dimensionality.
    idx, _ = train_test_split(
        np.arange(len(y_full)), train_size=N_SUBSAMPLE, stratify=y_full, random_state=seed
    )
    X, y = X_full[idx], y_full[idx]

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_qubits, random_state=seed)),
    ])
    X_pca = pipe.fit_transform(X)
    X_scaled = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit_transform(X_pca)
    return X_scaled, y


if __name__ == "__main__":
    for n_qubits in (4, 6):
        X, y = load_pca_reduced("miniboone", n_qubits=n_qubits)
        print("miniboone", n_qubits, X.shape, "class balance:", np.mean(y == 1))
