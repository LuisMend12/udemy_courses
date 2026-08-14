#pragma once

#include <vector>
#include <cstddef>

// Dense square-friendly matrix of doubles used by the Cholesky backend.
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols, double init = 0.0);
    explicit Matrix(const std::vector<std::vector<double>>& data);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    double& operator()(std::size_t i, std::size_t j);
    double operator()(std::size_t i, std::size_t j) const;

    Matrix transpose() const;
    Matrix multiply(const Matrix& other) const;

    bool isSquare() const;
    // Symmetric within an absolute tolerance (handles floating point input).
    bool isSymmetric(double tol = 1e-9) const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};
