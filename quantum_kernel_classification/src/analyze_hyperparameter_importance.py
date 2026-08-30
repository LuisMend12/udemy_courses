"""Which hyperparameter actually matters most for the quantum kernel:
qubit count / PCA dimensionality (n_qubits), feature-map repetitions (R), or
SVM regularization (C)? Runs one Optuna study per domain (synthetic,
miniboone) over all three jointly, then ranks them with Optuna's fANOVA
importance evaluator (optuna.importance.get_param_importances).

This extends optimize_quantum_kernel.py (which only searched R and C at a
fixed n_qubits per dataset) by also letting the search pick between the two
PCA/qubit-count variants already used in the paper (4 vs 6), so n_qubits
competes on equal footing with R and C for "most important."

Run:
    python analyze_hyperparameter_importance.py
Writes:
    ../results/hyperparameter_importance.json
"""
import json
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
N_TRIALS = 80


def get_domain_splits(domain: str):
    """{n_qubits: (X, y, train_idx, test_idx)} for the two dataset variants
    (n_qubits=4 and 6) already used for this domain in run_experiments.py."""
    if domain == "synthetic":
        raw = {
            4: generate_quantum_advantage_dataset(n_qubits=4, n_samples=200, gamma=0.3, seed=SEED)[:2],
            6: generate_quantum_advantage_dataset(n_qubits=6, n_samples=200, gamma=0.2, seed=SEED)[:2],
        }
    elif domain == "miniboone":
        raw = {
            4: load_pca_reduced("miniboone", n_qubits=4, seed=SEED),
            6: load_pca_reduced("miniboone", n_qubits=6, seed=SEED),
        }
    else:
        raise ValueError(domain)

    splits = {}
    for n_qubits, (X, y) in raw.items():
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED
        )
        splits[n_qubits] = (X, y, train_idx, test_idx)
    return splits


def make_objective(splits):
    def objective(trial):
        n_qubits = trial.suggest_categorical("n_qubits", sorted(splits.keys()))
        reps = trial.suggest_int("reps", 1, 4)
        C = trial.suggest_float("C", 0.01, 300.0, log=True)
        X, y, train_idx, test_idx = splits[n_qubits]
        K = normalize_kernel(quantum_kernel_matrix(X, X, n_qubits=n_qubits, reps=reps))
        res = cv_select_and_test(K, y, train_idx, test_idx, C_grid=[C], seed=SEED)
        return res["cv_acc"]
    return objective


def analyze_domain(domain: str):
    splits = get_domain_splits(domain)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(splits), n_trials=N_TRIALS)

    importances = optuna.importance.get_param_importances(study)
    ranked = ", ".join(f"{k}={v:.3f}" for k, v in importances.items())
    print(f"[{domain}] fANOVA importance (sums to 1): {ranked}")
    print(f"[{domain}] best: {study.best_params} cv_acc={study.best_value:.3f} "
          f"({N_TRIALS} trials)")

    return {
        "domain": domain,
        "importances": {k: float(v) for k, v in importances.items()},
        "best_params": study.best_params,
        "best_cv_acc": study.best_value,
        "n_trials": N_TRIALS,
    }


def main():
    results = [analyze_domain("synthetic"), analyze_domain("miniboone")]
    out_path = RESULTS_DIR / "hyperparameter_importance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nWrote", out_path)
    return results


if __name__ == "__main__":
    main()
