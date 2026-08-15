"""Main experiment driver: benchmark table (accuracy + kernel-target
alignment), learning curves, and a 2-qubit decision-boundary visualization.
Writes CSVs to ../results and PNGs to ../figures."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from quantum_kernel import quantum_kernel_matrix, kernel_target_alignment
from kernels import classical_kernel_fn, normalize_kernel
from model_selection import cv_select_and_test
from synthetic_data import generate_quantum_advantage_dataset
from datasets import load_pca_reduced

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 0
C_GRID = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
RBF_GAMMA_GRID = [0.02, 0.05, 0.1, 0.2, 0.5]
POLY_DEGREE_GRID = [2, 3]


def evaluate_all_kernels(X, y, n_qubits, dataset_name, reps=2, seed=SEED):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )
    rows = []

    def _run(kernel_label, K_raw, hyperparams):
        t0 = time.time()
        K = normalize_kernel(K_raw)
        res = cv_select_and_test(K, y, train_idx, test_idx, C_GRID, seed=seed)
        align = kernel_target_alignment(K[np.ix_(train_idx, train_idx)], y[train_idx])
        dt = time.time() - t0
        print(f"  [{dataset_name}] {kernel_label} {hyperparams} "
              f"cv_acc={res['cv_acc']:.3f} test_acc={res['test_acc']:.3f} "
              f"align={align:.3f} ({dt:.1f}s)", flush=True)
        row = {
            "dataset": dataset_name, "kernel": kernel_label,
            "hyperparams": {**hyperparams, "C": res["best_C"]}, "cv_acc": res["cv_acc"],
            "test_acc": res["test_acc"], "alignment": align, "seconds": dt,
        }
        return row

    # --- quantum kernel ---
    K_q = quantum_kernel_matrix(X, X, n_qubits=n_qubits, reps=reps)
    rows.append(_run("quantum(ZZ,full,reps=%d)" % reps, K_q, {}))

    # --- classical kernels ---
    for gamma in RBF_GAMMA_GRID:
        K = classical_kernel_fn("rbf", gamma=gamma)(X, X)
        rows.append(_run("rbf", K, {"gamma": gamma}))

    for degree in POLY_DEGREE_GRID:
        K = classical_kernel_fn("poly", degree=degree, gamma=1.0)(X, X)
        rows.append(_run("poly", K, {"degree": degree}))

    K = classical_kernel_fn("linear")(X, X)
    rows.append(_run("linear", K, {}))

    # collapse gamma/degree sweeps to the single best-by-CV row per kernel family
    best_by_family = {}
    for r in rows:
        fam = r["kernel"] if r["kernel"].startswith("quantum") else r["kernel"]
        key = "rbf" if fam == "rbf" else ("poly" if fam == "poly" else fam)
        if key not in best_by_family or r["cv_acc"] > best_by_family[key]["cv_acc"]:
            best_by_family[key] = r
    return list(best_by_family.values()), (train_idx, test_idx)


def run_benchmark():
    all_rows = []

    X, y, _ = generate_quantum_advantage_dataset(n_qubits=4, n_samples=200, gamma=0.3, seed=SEED)
    rows, _ = evaluate_all_kernels(X, y, n_qubits=4, dataset_name="synthetic_n4_gamma0.3")
    all_rows += rows

    X, y, _ = generate_quantum_advantage_dataset(n_qubits=6, n_samples=200, gamma=0.2, seed=SEED)
    rows, _ = evaluate_all_kernels(X, y, n_qubits=6, dataset_name="synthetic_n6_gamma0.2")
    all_rows += rows

    X, y = load_pca_reduced("miniboone", n_qubits=4, seed=SEED)
    rows, _ = evaluate_all_kernels(X, y, n_qubits=4, dataset_name="miniboone_pca4")
    all_rows += rows

    X, y = load_pca_reduced("miniboone", n_qubits=6, seed=SEED)
    rows, _ = evaluate_all_kernels(X, y, n_qubits=6, dataset_name="miniboone_pca6")
    all_rows += rows

    with open(RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)

    print(f"{'dataset':<24}{'kernel':<28}{'cv_acc':>8}{'test_acc':>10}{'alignment':>11}")
    for r in all_rows:
        print(f"{r['dataset']:<24}{r['kernel']:<28}{r['cv_acc']:>8.3f}{r['test_acc']:>10.3f}{r['alignment']:>11.3f}")

    return all_rows


if __name__ == "__main__":
    run_benchmark()
