"""Optuna search over the quantum kernel's hyperparameters (feature-map
repetitions R and SVM regularization C), on every dataset used in
run_experiments.py.

The paper (Limitations, item 2) fixes R=2 throughout and never tunes it --
this fills that gap. Reuses the exact same StratifiedKFold CV protocol as
model_selection.cv_select_and_test, so results are directly comparable to
results/benchmark.json.

Run:
    python optimize_quantum_kernel.py
Writes:
    ../results/quantum_kernel_hpo.json
"""
import json
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import train_test_split

from quantum_kernel import quantum_kernel_matrix
from kernels import normalize_kernel
from model_selection import cv_select_and_test
from synthetic_data import generate_quantum_advantage_dataset
from datasets import load_pca_reduced

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 0
N_TRIALS = 40
REPS_RANGE = (1, 4)      # R in {1,2,3,4}
C_RANGE = (0.01, 300.0)  # log-uniform, same order of magnitude as C_GRID


def make_objective(X, y, n_qubits, train_idx, test_idx):
    def objective(trial):
        reps = trial.suggest_int("reps", *REPS_RANGE)
        C = trial.suggest_float("C", *C_RANGE, log=True)
        K = normalize_kernel(quantum_kernel_matrix(X, X, n_qubits=n_qubits, reps=reps))
        res = cv_select_and_test(K, y, train_idx, test_idx, C_grid=[C], seed=SEED)
        trial.set_user_attr("test_acc", res["test_acc"])
        return res["cv_acc"]
    return objective


def optimize_dataset(X, y, n_qubits, dataset_name, baseline_reps=2):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(make_objective(X, y, n_qubits, train_idx, test_idx), n_trials=N_TRIALS)
    dt = time.time() - t0

    best = study.best_trial

    # Baseline: the fixed R=2 config the paper actually reports, for comparison.
    K_base = normalize_kernel(quantum_kernel_matrix(X, X, n_qubits=n_qubits, reps=baseline_reps))
    C_GRID = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    base_res = cv_select_and_test(K_base, y, train_idx, test_idx, C_GRID, seed=SEED)

    print(f"[{dataset_name}] baseline R={baseline_reps}: "
          f"cv_acc={base_res['cv_acc']:.3f} test_acc={base_res['test_acc']:.3f}")
    print(f"[{dataset_name}] optuna best: R={best.params['reps']} C={best.params['C']:.3g} "
          f"cv_acc={best.value:.3f} test_acc={best.user_attrs['test_acc']:.3f} "
          f"({N_TRIALS} trials, {dt:.1f}s)")

    return {
        "dataset": dataset_name,
        "n_qubits": n_qubits,
        "baseline": {"reps": baseline_reps, **base_res},
        "optuna_best": {
            "reps": best.params["reps"], "C": best.params["C"],
            "cv_acc": best.value, "test_acc": best.user_attrs["test_acc"],
        },
        "n_trials": N_TRIALS, "seconds": dt,
    }


def main():
    results = []

    X, y, _ = generate_quantum_advantage_dataset(n_qubits=4, n_samples=200, gamma=0.3, seed=SEED)
    results.append(optimize_dataset(X, y, n_qubits=4, dataset_name="synthetic_n4_gamma0.3"))

    X, y, _ = generate_quantum_advantage_dataset(n_qubits=6, n_samples=200, gamma=0.2, seed=SEED)
    results.append(optimize_dataset(X, y, n_qubits=6, dataset_name="synthetic_n6_gamma0.2"))

    X, y = load_pca_reduced("miniboone", n_qubits=4, seed=SEED)
    results.append(optimize_dataset(X, y, n_qubits=4, dataset_name="miniboone_pca4"))

    X, y = load_pca_reduced("miniboone", n_qubits=6, seed=SEED)
    results.append(optimize_dataset(X, y, n_qubits=6, dataset_name="miniboone_pca6"))

    with open(RESULTS_DIR / "quantum_kernel_hpo.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nWrote", RESULTS_DIR / "quantum_kernel_hpo.json")
    return results


if __name__ == "__main__":
    main()
