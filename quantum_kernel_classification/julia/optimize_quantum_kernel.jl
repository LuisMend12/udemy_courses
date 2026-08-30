# Native-Julia hyperparameter search for the quantum kernel (feature-map
# repetitions R and SVM regularization C), using QuantumKernel.jl for the
# kernel and LIBSVM.jl (a Julia wrapper around the same libsvm that
# scikit-learn's SVC calls) for the precomputed-kernel classifier -- an
# independent-language counterpart to optimize_quantum_kernel.py.
#
# Datasets and train/test splits are the exact ones used by the Python
# pipeline (seed=0), exported to julia/data/ by export_datasets_for_julia.py,
# so results are directly comparable to results/quantum_kernel_hpo.json.
#
# Search strategy: random search (Optuna's TPE isn't available in pure
# Julia without extra package installs; for this low-dimensional, smooth
# 2-parameter space -- R in {1,2,3,4}, C log-uniform -- random search over
# the same number of trials is a fair, dependency-free stand-in).
#
# Run:
#     julia optimize_quantum_kernel.jl
# Writes:
#     ../results/quantum_kernel_hpo_julia.json

include("QuantumKernel.jl")
using .QuantumKernel
using LIBSVM
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics
using JSON

const SEED = 0
const N_TRIALS = 40
const REPS_RANGE = (1, 4)
const C_RANGE = (0.01, 300.0)      # log-uniform
const C_GRID = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]   # baseline grid, matches Python

const DATA_DIR = joinpath(@__DIR__, "data")
const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

normalize_kernel(K) = K ./ (sqrt.(diag(K)) * sqrt.(diag(K))')

function load_dataset(name::String)
    X_py = readdlm(joinpath(DATA_DIR, "$(name)_X.csv"), ',')     # (n_samples, n_qubits)
    y = vec(Int.(readdlm(joinpath(DATA_DIR, "$(name)_y.csv"), ',')))
    train_idx = vec(Int.(readdlm(joinpath(DATA_DIR, "$(name)_train_idx.csv"), ','))) .+ 1
    test_idx = vec(Int.(readdlm(joinpath(DATA_DIR, "$(name)_test_idx.csv"), ','))) .+ 1
    X = permutedims(X_py)   # -> (n_qubits, n_samples) for QuantumKernel's convention
    return X, y, train_idx, test_idx
end

"""Stratified k-fold assignment (1..n_folds) for each element of y, computed
independently per class so fold sizes stay balanced across classes."""
function stratified_folds(y::AbstractVector{<:Integer}, n_folds::Int, rng::AbstractRNG)
    fold_of = Vector{Int}(undef, length(y))
    for class in unique(y)
        idx = findall(==(class), y)
        shuffle!(rng, idx)
        for (j, i) in enumerate(idx)
            fold_of[i] = ((j - 1) % n_folds) + 1
        end
    end
    return fold_of
end

"""CV-select-and-test for one fixed C, mirroring model_selection.cv_select_and_test
with C_grid=[C]: K-fold CV accuracy on train_idx, then refit on all of
train_idx and evaluate once on test_idx."""
function cv_and_test(K_full::AbstractMatrix, y::AbstractVector{<:Integer},
                      train_idx::Vector{Int}, test_idx::Vector{Int}, C::Real;
                      n_folds::Int = 5, seed::Int = SEED)
    y_train = y[train_idx]
    rng = MersenneTwister(seed)
    fold_of = stratified_folds(y_train, n_folds, rng)

    accs = Float64[]
    for fold in 1:n_folds
        val_mask = fold_of .== fold
        tr_idx = train_idx[.!val_mask]
        val_idx = train_idx[val_mask]

        K_tr = K_full[tr_idx, tr_idx]
        K_val = K_full[val_idx, tr_idx]     # (n_val, n_tr)

        model = svmtrain(K_tr, y[tr_idx]; svmtype = SVC, kernel = LIBSVM.Kernel.Precomputed, cost = Float64(C))
        pred, _ = svmpredict(model, permutedims(K_val))
        push!(accs, mean(pred .== y[val_idx]))
    end
    cv_acc = mean(accs)

    K_train_full = K_full[train_idx, train_idx]
    K_test = K_full[test_idx, train_idx]    # (n_test, n_train)
    model = svmtrain(K_train_full, y_train; svmtype = SVC, kernel = LIBSVM.Kernel.Precomputed, cost = Float64(C))
    pred, _ = svmpredict(model, permutedims(K_test))
    test_acc = mean(pred .== y[test_idx])

    return (cv_acc = cv_acc, test_acc = test_acc)
end

function grid_select_and_test(K_full, y, train_idx, test_idx, C_grid; seed = SEED)
    best_C, best_cv = C_grid[1], -Inf
    for C in C_grid
        r = cv_and_test(K_full, y, train_idx, test_idx, C; seed = seed)
        if r.cv_acc > best_cv
            best_cv, best_C = r.cv_acc, C
        end
    end
    final = cv_and_test(K_full, y, train_idx, test_idx, best_C; seed = seed)
    return (C = best_C, cv_acc = best_cv, test_acc = final.test_acc)
end

function random_search(X, y, n_qubits, train_idx, test_idx; n_trials = N_TRIALS, seed = SEED)
    rng = MersenneTwister(seed)
    log_lo, log_hi = log(C_RANGE[1]), log(C_RANGE[2])

    best = (reps = 0, C = 0.0, cv_acc = -Inf, test_acc = -Inf)
    for _ in 1:n_trials
        reps = rand(rng, REPS_RANGE[1]:REPS_RANGE[2])
        C = exp(log_lo + rand(rng) * (log_hi - log_lo))
        K = normalize_kernel(feature_map_statevectors_to_kernel(X, n_qubits, reps))
        r = cv_and_test(K, y, train_idx, test_idx, C; seed = seed)
        if r.cv_acc > best.cv_acc
            best = (reps = reps, C = C, cv_acc = r.cv_acc, test_acc = r.test_acc)
        end
    end
    return best
end

feature_map_statevectors_to_kernel(X, n_qubits, reps) = quantum_kernel_matrix(X, X, n_qubits; reps = reps)

function optimize_dataset(name::String; baseline_reps::Int = 2)
    X, y, train_idx, test_idx = load_dataset(name)
    n_qubits = size(X, 1)

    t0 = time()
    best = random_search(X, y, n_qubits, train_idx, test_idx)
    dt = time() - t0

    K_base = normalize_kernel(quantum_kernel_matrix(X, X, n_qubits; reps = baseline_reps))
    base = grid_select_and_test(K_base, y, train_idx, test_idx, C_GRID)

    println("[$name] baseline R=$baseline_reps: cv_acc=$(round(base.cv_acc, digits=3)) test_acc=$(round(base.test_acc, digits=3))")
    println("[$name] julia random search best: R=$(best.reps) C=$(round(best.C, sigdigits=3)) " *
            "cv_acc=$(round(best.cv_acc, digits=3)) test_acc=$(round(best.test_acc, digits=3)) " *
            "($N_TRIALS trials, $(round(dt, digits=1))s)")

    return Dict(
        "dataset" => name, "n_qubits" => n_qubits,
        "baseline" => Dict("reps" => baseline_reps, "C" => base.C, "cv_acc" => base.cv_acc, "test_acc" => base.test_acc),
        "julia_best" => Dict("reps" => best.reps, "C" => best.C, "cv_acc" => best.cv_acc, "test_acc" => best.test_acc),
        "n_trials" => N_TRIALS, "seconds" => dt,
    )
end

function main()
    names = ["synthetic_n4_gamma0.3", "synthetic_n6_gamma0.2", "miniboone_pca4", "miniboone_pca6"]
    results = [optimize_dataset(n) for n in names]

    open(joinpath(RESULTS_DIR, "quantum_kernel_hpo_julia.json"), "w") do io
        JSON.print(io, results, 2)
    end
    println("\nWrote ", joinpath(RESULTS_DIR, "quantum_kernel_hpo_julia.json"))
end

main()
