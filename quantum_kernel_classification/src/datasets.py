"""Real-dataset loaders: PCA-reduce to n_qubits dims, scale to [0, 2*pi) for
the quantum feature map (classical kernels are also fed this representation,
so the comparison isolates the kernel choice rather than confounding it with
different preprocessing)."""
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline


def load_pca_reduced(name: str, n_qubits: int, seed: int = 0):
    if name == "wine":
        data = load_wine()
        y = np.where(data.target == 0, 1.0, -1.0)  # class 0 vs rest (roughly balanced-ish)
    elif name == "breast_cancer":
        data = load_breast_cancer()
        y = np.where(data.target == 1, 1.0, -1.0)
    else:
        raise ValueError(name)

    X = data.data
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_qubits, random_state=seed)),
    ])
    X_pca = pipe.fit_transform(X)
    X_scaled = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit_transform(X_pca)
    return X_scaled, y


if __name__ == "__main__":
    for name in ("wine", "breast_cancer"):
        X, y = load_pca_reduced(name, n_qubits=4)
        print(name, X.shape, "class balance:", np.mean(y == 1))
