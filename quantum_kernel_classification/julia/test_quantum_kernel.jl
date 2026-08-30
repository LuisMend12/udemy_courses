"""
Sanity checks for QuantumKernel.jl, mirroring test_quantum_kernel.py:
  1. Brute-force cross-check: rebuild the ZZ feature-map circuit with explicit
     dense matrices (kron'd Hadamards + explicitly-looped diagonal phase) for
     small n, and compare against the vectorized/FWHT simulator bit-for-bit.
  2. Physical sanity: statevectors have unit norm (circuit is unitary).
  3. Kernel sanity: K is symmetric, diagonal ~1, and PSD (up to float noise).
"""
include("QuantumKernel.jl")
using .QuantumKernel
using LinearAlgebra
using Random

const H2 = [1.0 1.0; 1.0 -1.0] ./ sqrt(2.0)

function brute_force_statevector(x::Vector{Float64}, n_qubits::Int, reps::Int)
    Hn = H2
    for _ in 1:(n_qubits - 1)
        Hn = kron(Hn, H2)
    end

    dim = 1 << n_qubits
    theta = zeros(Float64, dim)
    for z in 0:dim-1
        bits = [(z >> i) & 1 for i in 0:n_qubits-1]
        s = [1 - 2b for b in bits]
        val = sum(x[i + 1] * s[i + 1] for i in 0:n_qubits-1)
        for i in 0:n_qubits-1
            for j in (i + 1):n_qubits-1
                val += (pi - x[i + 1]) * (pi - x[j + 1]) * s[i + 1] * s[j + 1]
            end
        end
        theta[z + 1] = val
    end
    D = Diagonal(cis.(theta))

    state = zeros(ComplexF64, dim)
    state[1] = 1.0
    for _ in 1:reps
        state = Hn * state
        state = D * state
    end
    return state
end

function test_matches_brute_force()
    rng = MersenneTwister(0)
    for n_qubits in (1, 2, 3, 4)
        x = rand(rng, n_qubits) .* (2 * pi)
        for reps in (1, 2, 3)
            fast = feature_map_statevectors(reshape(x, n_qubits, 1), n_qubits; reps = reps)[:, 1]
            slow = brute_force_statevector(x, n_qubits, reps)
            err = maximum(abs.(fast .- slow))
            @assert err < 1e-10 "mismatch n=$n_qubits reps=$reps: err=$err"
        end
    end
    println("PASS: FWHT simulator matches brute-force dense-matrix construction")
end

function test_unitary_norm()
    rng = MersenneTwister(1)
    X = rand(rng, 5, 50) .* (2 * pi)   # (n_qubits=5, n_samples=50)
    psi = feature_map_statevectors(X, 5; reps = 2)
    norms = vec(sum(abs.(psi) .^ 2, dims = 1))
    @assert all(isapprox.(norms, 1.0; atol = 1e-9)) norms
    println("PASS: all statevectors have unit norm (circuit is unitary)")
end

function test_kernel_properties()
    rng = MersenneTwister(2)
    X = rand(rng, 4, 30) .* (2 * pi)   # (n_qubits=4, n_samples=30)
    K = quantum_kernel_matrix(X, X, 4; reps = 2)
    @assert isapprox(K, K'; atol = 1e-9) "kernel not symmetric"
    @assert all(isapprox.(diag(K), 1.0; atol = 1e-9)) "self-kernel entries not 1"
    ev = eigvals(Symmetric(K))
    @assert minimum(ev) > -1e-8 "kernel not PSD, min eigenvalue=$(minimum(ev))"
    println("PASS: kernel matrix is symmetric, PSD, with unit diagonal")
end

function test_pm1_matrix()
    S = pm1_matrix(3)
    @assert size(S) == (8, 3)
    @assert Set(unique(S)) == Set([-1, 1])
    println("PASS: +-1 basis matrix has correct shape/values")
end

function main()
    test_pm1_matrix()
    test_matches_brute_force()
    test_unitary_norm()
    test_kernel_properties()
    println()
    println("All sanity checks passed.")
end

main()
