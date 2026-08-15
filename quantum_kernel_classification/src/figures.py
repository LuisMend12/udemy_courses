"""Learning-curve and decision-boundary figures for the paper. Uses the
hyperparameters already selected by run_experiments.py's CV (results/benchmark.json)
so this script does not re-tune anything -- it only asks "how many samples are
needed to reach that fixed operating point," and the same fixed kernel/C is used
across all training sizes."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from quantum_kernel import quantum_kernel_matrix
from kernels import classical_kernel_fn, normalize_kernel
from synthetic_data import generate_quantum_advantage_dataset
from datasets import load_pca_reduced

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEED = 0

PALETTE = {
    "quantum": "#4C72B0",
    "rbf": "#DD8452",
    "poly": "#55A868",
    "linear": "#8172B2",
}
LABELS = {
    "quantum": "Quantum (ZZ)",
    "rbf": "RBF",
    "poly": "Polynomial",
    "linear": "Linear",
}


def _kernel_family(name: str) -> str:
    return "quantum" if name.startswith("quantum") else name


def _full_kernel_matrix(X, family, hp, n_qubits):
    if family == "quantum":
        return normalize_kernel(quantum_kernel_matrix(X, X, n_qubits=n_qubits, reps=2))
    if family == "rbf":
        return normalize_kernel(classical_kernel_fn("rbf", gamma=hp["gamma"])(X, X))
    if family == "poly":
        return normalize_kernel(classical_kernel_fn("poly", degree=hp["degree"], gamma=1.0)(X, X))
    if family == "linear":
        return normalize_kernel(classical_kernel_fn("linear")(X, X))
    raise ValueError(family)


def load_best_hyperparams():
    with open(RESULTS_DIR / "benchmark.json") as f:
        rows = json.load(f)
    best = {}
    for r in rows:
        best[(r["dataset"], _kernel_family(r["kernel"]))] = r["hyperparams"]
    return best


def learning_curve(X, y, n_qubits, dataset_name, best_hp, train_sizes, n_repeats=8, seed=SEED):
    train_idx_full, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )
    K_by_family = {
        fam: _full_kernel_matrix(X, fam, best_hp[(dataset_name, fam)], n_qubits)
        for fam in ("quantum", "rbf", "poly", "linear")
    }
    curves = {fam: [] for fam in K_by_family}
    rng = np.random.default_rng(seed)

    for n_train in train_sizes:
        for fam, K in K_by_family.items():
            C = best_hp[(dataset_name, fam)]["C"]
            accs = []
            for rep in range(n_repeats):
                sub_seed = int(rng.integers(0, 1_000_000))
                sub_idx, _ = train_test_split(
                    train_idx_full, train_size=n_train,
                    stratify=y[train_idx_full], random_state=sub_seed,
                )
                clf = SVC(kernel="precomputed", C=C, max_iter=200_000)
                clf.fit(K[np.ix_(sub_idx, sub_idx)], y[sub_idx])
                acc = clf.score(K[np.ix_(test_idx, sub_idx)], y[test_idx])
                accs.append(acc)
            curves[fam].append((np.mean(accs), np.std(accs)))
    return curves


def plot_learning_curve(curves, train_sizes, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for fam, vals in curves.items():
        means = np.array([v[0] for v in vals])
        stds = np.array([v[1] for v in vals])
        ax.plot(train_sizes, means, marker="o", label=LABELS[fam], color=PALETTE[fam])
        ax.fill_between(train_sizes, means - stds, means + stds, alpha=0.15, color=PALETTE[fam])
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.set_ylim(0.35, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def decision_boundary_figure(out_path, n_qubits=2, gamma=0.35, seed=SEED):
    X, y, _ = generate_quantum_advantage_dataset(n_qubits=n_qubits, n_samples=160, gamma=gamma, seed=seed)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)

    gx = np.linspace(0, 2 * np.pi, 60)
    gy = np.linspace(0, 2 * np.pi, 60)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)

    families = {
        "quantum": lambda: normalize_kernel(quantum_kernel_matrix(np.vstack([X, grid]), np.vstack([X, grid]), n_qubits=n_qubits, reps=2)),
        "rbf": lambda: normalize_kernel(classical_kernel_fn("rbf", gamma=0.5)(np.vstack([X, grid]), np.vstack([X, grid]))),
        "poly": lambda: normalize_kernel(classical_kernel_fn("poly", degree=3, gamma=1.0)(np.vstack([X, grid]), np.vstack([X, grid]))),
        "linear": lambda: normalize_kernel(classical_kernel_fn("linear")(np.vstack([X, grid]), np.vstack([X, grid]))),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    n_data = len(X)
    for ax, (fam, kfn) in zip(axes, families.items()):
        K_all = kfn()
        K_data = K_all[:n_data][:, :n_data]
        K_grid = K_all[n_data:][:, :n_data]
        clf = SVC(kernel="precomputed", C=10.0, max_iter=200_000)
        clf.fit(K_data[np.ix_(train_idx, train_idx)], y[train_idx])
        pred = clf.predict(K_grid[:, train_idx]).reshape(GX.shape)
        ax.contourf(GX, GY, pred, levels=[-1.5, 0, 1.5], colors=["#DDE7F5", "#FBE3D6"], alpha=0.9)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c=PALETTE["quantum"], s=14, label="+1", edgecolors="none")
        ax.scatter(X[y == -1, 0], X[y == -1, 1], c=PALETTE["rbf"], s=14, label="-1", edgecolors="none")
        ax.set_title(LABELS[fam])
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Decision regions on the 2-qubit quantum-advantage dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    best_hp = load_best_hyperparams()

    X, y, _ = generate_quantum_advantage_dataset(n_qubits=4, n_samples=200, gamma=0.3, seed=SEED)
    sizes = [10, 20, 30, 50, 70, 90, 110, 130]
    curves = learning_curve(X, y, 4, "synthetic_n4_gamma0.3", best_hp, sizes)
    plot_learning_curve(curves, sizes, "Sample efficiency: engineered quantum-advantage data",
                         FIGURES_DIR / "learning_curve_synthetic.png")
    print("Saved learning_curve_synthetic.png")

    X, y = load_pca_reduced("wine", n_qubits=4, seed=SEED)
    sizes = [10, 20, 30, 50, 70, 90, 110]
    curves = learning_curve(X, y, 4, "wine_pca4", best_hp, sizes)
    plot_learning_curve(curves, sizes, "Sample efficiency: Wine (PCA-4)",
                         FIGURES_DIR / "learning_curve_wine.png")
    print("Saved learning_curve_wine.png")

    decision_boundary_figure(FIGURES_DIR / "decision_boundary.png")
    print("Saved decision_boundary.png")


if __name__ == "__main__":
    main()
