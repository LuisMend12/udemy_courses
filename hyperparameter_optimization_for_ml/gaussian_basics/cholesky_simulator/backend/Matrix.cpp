#include "Matrix.hpp"
#include <stdexcept>
#include <cmath>

Matrix::Matrix(std::size_t rows, std::size_t cols, double init)
    : rows_(rows), cols_(cols), data_(rows * cols, init) {}

Matrix::Matrix(const std::vector<std::vector<double>>& data) {
    rows_ = data.size();
    cols_ = rows_ == 0 ? 0 : data[0].size();
    data_.resize(rows_ * cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        if (data[i].size() != cols_)
            throw std::invalid_argument("All matrix rows must have the same length.");
        for (std::size_t j = 0; j < cols_; ++j)
            (*this)(i, j) = data[i][j];
    }
}

double& Matrix::operator()(std::size_t i, std::size_t j) {
    return data_[i * cols_ + j];
}

double Matrix::operator()(std::size_t i, std::size_t j) const {
    return data_[i * cols_ + j];
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = 0; j < cols_; ++j)
            result(j, i) = (*this)(i, j);
    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("Matrix dimension mismatch in multiply().");
    Matrix result(rows_, other.cols_);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < cols_; ++k)
                sum += (*this)(i, k) * other(k, j);
            result(i, j) = sum;
        }
    return result;
}

bool Matrix::isSquare() const {
    return rows_ == cols_;
}

bool Matrix::isSymmetric(double tol) const {
    if (!isSquare())
        return false;
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = i + 1; j < cols_; ++j)
            if (std::fabs((*this)(i, j) - (*this)(j, i)) > tol)
                return false;
    return true;
}
