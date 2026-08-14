#pragma once

#include "Matrix.hpp"
#include <string>
#include <vector>

// Result of attempting a Cholesky factorization A = L * L^T.
struct CholeskyResult {
    bool success = false;
    std::string errorMessage;
    Matrix L{0, 0};
};

// Pure computation layer: no I/O, no console interaction.
// The frontend (frontend/main.cpp) is the only place that talks to the user.
class CholeskyDecomposer {
public:
    // Factorizes A into lower-triangular L such that A = L * L^T.
    // Fails (success = false) if A is not square, not symmetric, or not
    // positive definite (a diagonal pivot would require sqrt of a negative
    // number).
    static CholeskyResult decompose(const Matrix& A, double tol = 1e-9);

    // det(A) = product(L_ii)^2, valid only for a successful result.
    static double determinant(const CholeskyResult& result);

    // Solves A x = b via L y = b (forward substitution) followed by
    // L^T x = y (back substitution). Valid only for a successful result.
    static std::vector<double> solve(const CholeskyResult& result,
                                      const std::vector<double>& b);

    // Recomputes L * L^T and compares against A within tolerance; used to
    // let the frontend show a correctness check on the result.
    static bool verify(const Matrix& A, const Matrix& L, double tol = 1e-6);
};
