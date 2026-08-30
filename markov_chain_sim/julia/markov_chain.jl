#!/usr/bin/env julia
# Discrete-time, finite-state Markov chain simulator.
#
# Simulates N independent chains stepping through a user-supplied (or
# randomly generated) row-stochastic transition matrix P for a fixed number
# of steps, writes every chain's full state trajectory to CSV, and as a
# sanity check compares the chains' empirical final-state distribution
# against the analytic stationary distribution of P (found by power
# iteration on the distribution itself, independent of the Monte Carlo
# simulation).
#
# Run:
#   julia markov_chain.jl --states 5 --chains 500 --steps 200 --init random --seed 0 --out trajectories.csv
#   julia markov_chain.jl --matrix my_transition_matrix.csv --chains 500 --steps 200 --out trajectories.csv
#
# Follows the same manual-CLI-parsing / CSV-output / built-in-sanity-check
# style as brownian_motion_sim/julia/brownian_motion.jl.

using Random
using LinearAlgebra
using Printf
using DelimitedFiles

mutable struct Config
    n_states::Int
    chains::Int
    steps::Int
    init::String          # "0", "3", ... (a state index) or "random"
    matrix_path::String   # "" -> generate a random ergodic matrix
    seed::UInt64
    out::String
end

Config() = Config(5, 200, 500, "0", "", rand(RandomDevice(), UInt64), "markov_trajectories.csv")

function print_usage(prog::AbstractString)
    println("""
    Usage: $prog [options]
      --states N      number of states, used only when --matrix is not given (default 5)
      --matrix FILE   CSV of a row-stochastic transition matrix (overrides --states)
      --chains N      number of independent chains to simulate (default 200)
      --steps N       number of time steps per chain (default 500)
      --init X        initial state: a 0-indexed integer, or "random" (default 0)
      --seed N        RNG seed (default random)
      --out FILE      output CSV path for trajectories (default markov_trajectories.csv)
      --help          show this message""")
end

function parse_config(args::Vector{String})
    cfg = Config()
    prog = PROGRAM_FILE == "" ? "markov_chain.jl" : basename(PROGRAM_FILE)

    i = 1
    next_val = (name) -> begin
        if i + 1 > length(args)
            println(stderr, "Missing value for $name")
            exit(1)
        end
        i += 1
        args[i]
    end

    while i <= length(args)
        arg = args[i]
        if arg == "--states"
            cfg.n_states = parse(Int, next_val("--states"))
        elseif arg == "--matrix"
            cfg.matrix_path = next_val("--matrix")
        elseif arg == "--chains"
            cfg.chains = parse(Int, next_val("--chains"))
        elseif arg == "--steps"
            cfg.steps = parse(Int, next_val("--steps"))
        elseif arg == "--init"
            cfg.init = next_val("--init")
        elseif arg == "--seed"
            cfg.seed = parse(UInt64, next_val("--seed"))
        elseif arg == "--out"
            cfg.out = next_val("--out")
        elseif arg == "--help" || arg == "-h"
            print_usage(prog)
            exit(0)
        else
            println(stderr, "Unknown argument: $arg")
            print_usage(prog)
            exit(1)
        end
        i += 1
    end

    if cfg.matrix_path == "" && cfg.n_states < 1
        println(stderr, "--states must be positive")
        exit(1)
    end
    if cfg.chains <= 0 || cfg.steps <= 0
        println(stderr, "chains/steps must be positive")
        exit(1)
    end
    return cfg
end

"""Random row-stochastic n x n matrix: each row is an independent Dirichlet(1,...,1)
draw (normalized Exp(1) samples), so almost surely every entry is positive
-- irreducible and aperiodic with probability 1, i.e. a valid ergodic chain."""
function random_transition_matrix(n::Int, rng::AbstractRNG)
    P = -log.(rand(rng, n, n))    # Exp(1) samples via inverse CDF
    P ./= sum(P, dims = 2)        # normalize each row to sum to 1
    return P
end

function load_transition_matrix(path::String)
    P = readdlm(path, ',')
    n, m = size(P)
    n == m || throw(ArgumentError("transition matrix must be square, got $(n)x$(m)"))
    row_sums = vec(sum(P, dims = 2))
    all(isapprox.(row_sums, 1.0; atol = 1e-6)) || throw(ArgumentError(
        "each row of the transition matrix must sum to 1, got sums $row_sums"))
    return P
end

"""Stationary distribution via power iteration on the distribution vector:
pi_{t+1} = pi_t * P, starting from uniform, until it stops moving."""
function stationary_distribution(P::AbstractMatrix; tol::Float64 = 1e-14, max_iters::Int = 200_000)
    n = size(P, 1)
    pi_vec = fill(1.0 / n, n)
    for _ in 1:max_iters
        pi_next = pi_vec' * P
        pi_next = vec(pi_next)
        if maximum(abs.(pi_next .- pi_vec)) < tol
            return pi_next
        end
        pi_vec = pi_next
    end
    return pi_vec
end

total_variation_distance(p::AbstractVector, q::AbstractVector) = 0.5 * sum(abs.(p .- q))

"""Sample the next state given current state `s` (1-indexed) and cumulative
row `cum` = cumsum(P[s, :])."""
function sample_next(cum::AbstractVector{<:Real}, rng::AbstractRNG)
    u = rand(rng)
    # clamp guards against u landing fractionally above a cumsum that should
    # be exactly 1.0 but drifted slightly below it due to float rounding
    return min(searchsortedfirst(cum, u), length(cum))
end

function simulate(cfg::Config)
    rng = Xoshiro(cfg.seed)

    P = cfg.matrix_path == "" ? random_transition_matrix(cfg.n_states, rng) : load_transition_matrix(cfg.matrix_path)
    n_states = size(P, 1)

    if cfg.matrix_path == ""
        mat_path = joinpath(dirname(cfg.out) == "" ? "." : dirname(cfg.out), "transition_matrix.csv")
        writedlm(mat_path, P, ',')
        println("Generated random ", n_states, "x", n_states, " transition matrix -> ", mat_path)
    end

    cumP = [cumsum(P[s, :]) for s in 1:n_states]   # cumP[s][j] = P(state <= j | state == s)

    state = Vector{Int}(undef, cfg.chains)
    if cfg.init == "random"
        for c in 1:cfg.chains
            state[c] = rand(rng, 1:n_states)
        end
    else
        s0 = parse(Int, cfg.init) + 1   # CLI/CSV convention is 0-indexed
        (1 <= s0 <= n_states) || throw(ArgumentError("--init state out of range [0, $(n_states-1)]"))
        fill!(state, s0)
    end

    io = open(cfg.out, "w")
    println(io, "step,chain,state")
    for c in 1:cfg.chains
        println(io, 0, ",", c - 1, ",", state[c] - 1)
    end

    for step in 1:cfg.steps
        for c in 1:cfg.chains
            state[c] = sample_next(cumP[state[c]], rng)
            println(io, step, ",", c - 1, ",", state[c] - 1)
        end
    end
    close(io)

    # Sanity check: empirical distribution of final states across chains vs.
    # the analytic stationary distribution of P (computed independently of
    # the Monte Carlo simulation above).
    empirical = zeros(Float64, n_states)
    for c in 1:cfg.chains
        empirical[state[c]] += 1
    end
    empirical ./= cfg.chains
    pi_star = stationary_distribution(P)
    tv = total_variation_distance(empirical, pi_star)

    println("Simulated ", cfg.chains, " chains, ", cfg.steps, " steps, ",
            n_states, " states, seed=", cfg.seed)
    @printf("Stationary distribution (power iteration): %s\n", string(round.(pi_star, digits = 4)))
    @printf("Empirical final-state distribution:         %s\n", string(round.(empirical, digits = 4)))
    @printf("Total variation distance (empirical vs. stationary): %.4f\n", tv)
    println("Wrote trajectories to ", cfg.out)
end

function main()
    cfg = parse_config(ARGS)
    simulate(cfg)
end

main()
