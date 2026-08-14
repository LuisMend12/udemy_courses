#include "Cholesky.hpp"
#include <cmath>
#include <stdexcept>

CholeskyResult CholeskyDecomposer::decompose(const Matrix& A, double tol) {
    CholeskyResult result;

    if (!A.isSquare()) {
        result.errorMessage = "Matrix must be square.";
        return result;
    }
    if (!A.isSymmetric(tol)) {
        result.errorMessage = "Matrix must be symmetric (A must equal A^T).";
        return result;
    }

    const std::size_t n = A.rows();
    Matrix L(n, n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k)
                sum += L(i, k) * L(j, k);

            if (i == j) {
                double diag = A(i, i) - sum;
                if (diag <= tol) {
                    result.errorMessage =
                        "Matrix is not positive definite (non-positive pivot "
                        "at row " + std::to_string(i) + ").";
                    return result;
                }
                L(i, j) = std::sqrt(diag);
            } else {
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }
    }

    result.success = true;
    result.L = L;
    return result;
}

double CholeskyDecomposer::determinant(const CholeskyResult& result) {
    if (!result.success)
        throw std::invalid_argument("Cannot compute determinant of a failed decomposition.");
    double diagProduct = 1.0;
    for (std::size_t i = 0; i < result.L.rows(); ++i)
        diagProduct *= result.L(i, i);
    return diagProduct * diagProduct;
}

std::vector<double> CholeskyDecomposer::solve(const CholeskyResult& result,
                                               const std::vector<double>& b) {
    if (!result.success)
        throw std::invalid_argument("Cannot solve using a failed decomposition.");

    const Matrix& L = result.L;
    const std::size_t n = L.rows();
    if (b.size() != n)
        throw std::invalid_argument("Right-hand side vector b has the wrong size.");

    // Forward substitution: L y = b
    std::vector<double> y(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = b[i];
        for (std::size_t k = 0; k < i; ++k)
            sum -= L(i, k) * y[k];
        y[i] = sum / L(i, i);
    }

    // Back substitution: L^T x = y
    std::vector<double> x(n, 0.0);
    for (std::size_t ii = 0; ii < n; ++ii) {
        std::size_t i = n - 1 - ii;
        double sum = y[i];
        for (std::size_t k = i + 1; k < n; ++k)
            sum -= L(k, i) * x[k];
        x[i] = sum / L(i, i);
    }

    return x;
}

bool CholeskyDecomposer::verify(const Matrix& A, const Matrix& L, double tol) {
    Matrix product = L.multiply(L.transpose());
    if (product.rows() != A.rows() || product.cols() != A.cols())
        return false;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            if (std::fabs(product(i, j) - A(i, j)) > tol)
                return false;
    return true;
}
