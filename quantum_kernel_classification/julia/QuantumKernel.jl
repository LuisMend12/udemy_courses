"""
From-scratch exact simulation of the Havlicek et al. (2019) ZZ feature-map
quantum kernel, implemented without any quantum-computing library.

Circuit (per repetition):
    |0>^n --H^{⊗n}--> uniform superposition
                    --diag(exp(i*theta(x)))--> phase-encoded state

theta(z, x) = sum_i x_i * (-1)^{z_i}
            + sum_{i<j} (pi - x_i)(pi - x_j) * (-1)^{z_i} * (-1)^{z_j}

which is the "full entanglement" Pauli-ZZ feature map. Repeated `reps` times.

Because the diagonal unitary is diagonal in the computational basis, the
whole circuit reduces to alternating a batched fast Walsh-Hadamard transform
with an elementwise complex phase multiply -- no gate-by-gate simulation or
quantum library is needed, and it is exact (not a Monte-Carlo estimate).

The fidelity kernel is k(x, x') = |<phi(x)|phi(x')>|^2.

Port of quantum_kernel.py. Samples are stored column-wise (Julia is
column-major): X is (n_qubits, n_samples), statevectors are
(2^n_qubits, n_samples).
"""
module QuantumKernel

using LinearAlgebra
using Random

export pm1_matrix, feature_map_statevectors, quantum_kernel_matrix,
       kernel_target_alignment, random_haar_unitary

"""
    pm1_matrix(n_qubits) -> Matrix{Int}

Rows = all length-`n_qubits` +-1 vectors, i.e. (-1)^{z_i} for
z in 0:2^n_qubits-1. Row `z+1`, column `i+1` holds (-1)^{bit i of z}
(0-indexed bit i, i.e. `(z >> i) & 1`), shape (2^n_qubits, n_qubits).
"""
function pm1_matrix(n_qubits::Int)
    N = 1 << n_qubits
    S = Matrix{Int}(undef, N, n_qubits)
    for z in 0:N-1
        for i in 0:n_qubits-1
            bit = (z >> i) & 1
            S[z + 1, i + 1] = 1 - 2 * bit
        end
    end
    return S
end

"""
    fwht!(v)

In-place fast Walsh-Hadamard transform of a length-N=2^n vector, normalized
to equal H^{⊗n} applied under the bit convention idx = sum_i b_i * 2^i
(0-indexed bit i), the same convention used by `pm1_matrix`.
"""
function fwht!(v::AbstractVector{<:Number})
    N = length(v)
    h = 1
    while h < N
        i = 1
        while i <= N
            @inbounds for j in i:(i + h - 1)
                a = v[j]
                b = v[j + h]
                v[j] = a + b
                v[j + h] = a - b
            end
            i += 2h
        end
        h *= 2
    end
    v ./= sqrt(N)
    return v
end

"""
    hadamard_transform_batch!(state)

Apply H^{⊗n_qubits} to each column of `state` (shape (2^n_qubits, n_samples))
in place, where n_qubits = log2(size(state, 1)).
"""
function hadamard_transform_batch!(state::AbstractMatrix{<:Number})
    for k in axes(state, 2)
        fwht!(@view state[:, k])
    end
    return state
end

"""
    phase_angles(X, S) -> Matrix

theta(z, x) for every sample (column of X, shape (n_qubits, n_samples)) and
every basis state z (row of S, shape (2^n_qubits, n_qubits)). Returns a
(2^n_qubits, n_samples) matrix.

Uses sum_{i<j} S_i S_j u_i u_j = 0.5*[(S.u)^2 - sum_i u_i^2] with u = pi - x,
which follows from S_i^2 = 1, to avoid an O(n^2) loop over qubit pairs.
"""
function phase_angles(X::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real})
    linear = S * X                                    # (2^n, n_samples)
    u = pi .- X                                        # (n_qubits, n_samples)
    su = S * u                                         # (2^n, n_samples)
    quadratic = 0.5 .* (su .^ 2 .- sum(u .^ 2, dims = 1))  # broadcasts (1, n_samples)
    return linear .+ quadratic
end

"""
    feature_map_statevectors(X, n_qubits; reps=2) -> Matrix{ComplexF64}

Exact statevectors |phi(x)> for each column of X (shape (n_qubits,
n_samples)). Returns (2^n_qubits, n_samples) ComplexF64, one column per
sample.
"""
function feature_map_statevectors(X::AbstractMatrix{<:Real}, n_qubits::Int; reps::Int = 2)
    size(X, 1) == n_qubits || throw(ArgumentError(
        "expected $n_qubits features, got $(size(X, 1))"))
    S = Float64.(pm1_matrix(n_qubits))
    n_samples = size(X, 2)
    N = 1 << n_qubits
    state = zeros(ComplexF64, N, n_samples)
    state[1, :] .= 1.0   # |0...0>
    for _ in 1:reps
        hadamard_transform_batch!(state)
        theta = phase_angles(X, S)
        state .*= cis.(theta)
    end
    return state
end

"""
    quantum_kernel_matrix(X1, X2, n_qubits; reps=2) -> Matrix{Float64}

Fidelity kernel k(x,x') = |<phi(x)|phi(x')>|^2 between columns of X1 and X2
(each shape (n_qubits, n_samples)).
"""
function quantum_kernel_matrix(X1::AbstractMatrix{<:Real}, X2::AbstractMatrix{<:Real},
                                n_qubits::Int; reps::Int = 2)
    psi1 = feature_map_statevectors(X1, n_qubits; reps = reps)
    psi2 = X2 === X1 ? psi1 : feature_map_statevectors(X2, n_qubits; reps = reps)
    gram = psi1' * psi2
    return abs2.(gram)
end

"""
    kernel_target_alignment(K, y) -> Float64

Cristianini et al. (2001) kernel-target alignment, y in {-1, +1}.
"""
function kernel_target_alignment(K::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    yy = y * y'
    num = sum(K .* yy)
    den = sqrt(sum(K .* K) * sum(yy .* yy))
    return num / den
end

"""
    random_haar_unitary(dim, rng) -> Matrix{ComplexF64}

Haar-random unitary via QR decomposition of a complex Gaussian matrix
(Mezzadri, 2006).
"""
function random_haar_unitary(dim::Int, rng::AbstractRNG)
    z = (randn(rng, dim, dim) .+ im .* randn(rng, dim, dim)) ./ sqrt(2.0)
    F = qr(z)
    Q = Matrix(F.Q)
    d = diag(F.R)
    ph = d ./ abs.(d)
    return Q .* transpose(ph)   # correct Haar measure (Mezzadri, 2006)
end

end # module
