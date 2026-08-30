"""Export the exact datasets and train/test splits used in run_experiments.py
/ optimize_quantum_kernel.py to CSV, so optimize_quantum_kernel.jl can search
over the identical data (same seed=0) without reimplementing the dataset
generators (synthetic_data.py, datasets.py) in Julia.

Run:
    python export_datasets_for_julia.py
Writes, per dataset, to ../julia/data/<name>_{X,y,train_idx,test_idx}.csv
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from synthetic_data import generate_quantum_advantage_dataset
from datasets import load_pca_reduced

OUT_DIR = Path(__file__).resolve().parent.parent / "julia" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0


def export(X, y, n_qubits, name):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED
    )
    np.savetxt(OUT_DIR / f"{name}_X.csv", X, delimiter=",")
    np.savetxt(OUT_DIR / f"{name}_y.csv", y, delimiter=",", fmt="%d")
    np.savetxt(OUT_DIR / f"{name}_train_idx.csv", train_idx, delimiter=",", fmt="%d")
    np.savetxt(OUT_DIR / f"{name}_test_idx.csv", test_idx, delimiter=",", fmt="%d")
    print(f"exported {name}: X{X.shape} n_qubits={n_qubits} "
          f"train={len(train_idx)} test={len(test_idx)}")


def main():
    X, y, _ = generate_quantum_advantage_dataset(n_qubits=4, n_samples=200, gamma=0.3, seed=SEED)
    export(X, y, 4, "synthetic_n4_gamma0.3")

    X, y, _ = generate_quantum_advantage_dataset(n_qubits=6, n_samples=200, gamma=0.2, seed=SEED)
    export(X, y, 6, "synthetic_n6_gamma0.2")

    X, y = load_pca_reduced("miniboone", n_qubits=4, seed=SEED)
    export(X, y, 4, "miniboone_pca4")

    X, y = load_pca_reduced("miniboone", n_qubits=6, seed=SEED)
    export(X, y, 6, "miniboone_pca6")


if __name__ == "__main__":
    main()
