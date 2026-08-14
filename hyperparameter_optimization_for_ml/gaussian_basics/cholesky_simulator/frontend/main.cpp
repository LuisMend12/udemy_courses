// Interactive console frontend for the Cholesky Decomposition Simulator.
// All numerical work happens in the backend/ library; this file is only
// responsible for user interaction and formatting output.
#include "../backend/Matrix.hpp"
#include "../backend/Cholesky.hpp"

#include <iostream>
#include <iomanip>
#include <limits>
#include <optional>
#include <vector>

namespace {

void printMatrix(const Matrix& m, const std::string& label) {
    std::cout << label << ":\n";
    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j)
            std::cout << std::setw(10) << m(i, j);
        std::cout << "\n";
    }
    std::cout.unsetf(std::ios::fixed);
    std::cout << std::setprecision(6);
}

void printVector(const std::vector<double>& v, const std::string& label) {
    std::cout << label << ": [ ";
    std::cout << std::fixed << std::setprecision(4);
    for (double x : v)
        std::cout << x << " ";
    std::cout.unsetf(std::ios::fixed);
    std::cout << "]\n";
}

// Reads a whitespace/newline separated n x n matrix from stdin.
std::optional<Matrix> readMatrix() {
    std::size_t n = 0;
    std::cout << "Matrix size n (for an n x n matrix): ";
    if (!(std::cin >> n) || n == 0) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid size.\n";
        return std::nullopt;
    }

    std::vector<std::vector<double>> data(n, std::vector<double>(n));
    std::cout << "Enter the " << n << "x" << n
              << " matrix, row by row (values separated by spaces):\n";
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (!(std::cin >> data[i][j])) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input.\n";
                return std::nullopt;
            }
        }
    }
    return Matrix(data);
}

std::optional<std::vector<double>> readVector(std::size_t n) {
    std::vector<double> b(n);
    std::cout << "Enter the " << n << " entries of b (separated by spaces): ";
    for (std::size_t i = 0; i < n; ++i) {
        if (!(std::cin >> b[i])) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input.\n";
            return std::nullopt;
        }
    }
    return b;
}

Matrix exampleMatrix() {
    return Matrix({
        { 4, 12, -16 },
        { 12, 37, -43 },
        { -16, -43, 98 }
    });
}

void printMenu() {
    std::cout << "\n=== Cholesky Decomposition Simulator ===\n"
              << "1) Enter a new matrix\n"
              << "2) Check symmetric positive-definite\n"
              << "3) Compute Cholesky decomposition (L, L^T)\n"
              << "4) Solve Ax = b using the decomposition\n"
              << "5) Compute determinant via decomposition\n"
              << "6) Load example matrix\n"
              << "7) Exit\n"
              << "Select option: ";
}

} // namespace

int main() {
    std::optional<Matrix> A;
    CholeskyResult lastResult;

    std::cout << "Cholesky Decomposition Simulator\n"
              << "Backend: backend/Cholesky.cpp  |  Frontend: this console UI\n";

    while (true) {
        printMenu();
        int choice = 0;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Please enter a number from the menu.\n";
            continue;
        }

        switch (choice) {
            case 1: {
                if (auto m = readMatrix()) {
                    A = m;
                    lastResult = CholeskyResult{};
                    std::cout << "Matrix stored.\n";
                }
                break;
            }
            case 2: {
                if (!A) { std::cout << "No matrix loaded. Use option 1 or 6 first.\n"; break; }
                if (!A->isSquare()) {
                    std::cout << "Result: NOT valid (matrix is not square).\n";
                    break;
                }
                if (!A->isSymmetric()) {
                    std::cout << "Result: NOT symmetric positive definite (matrix is not symmetric).\n";
                    break;
                }
                auto trial = CholeskyDecomposer::decompose(*A);
                if (trial.success)
                    std::cout << "Result: symmetric positive definite. Cholesky decomposition exists.\n";
                else
                    std::cout << "Result: symmetric, but NOT positive definite (" << trial.errorMessage << ")\n";
                break;
            }
            case 3: {
                if (!A) { std::cout << "No matrix loaded. Use option 1 or 6 first.\n"; break; }
                lastResult = CholeskyDecomposer::decompose(*A);
                if (!lastResult.success) {
                    std::cout << "Decomposition failed: " << lastResult.errorMessage << "\n";
                    break;
                }
                printMatrix(lastResult.L, "L (lower triangular)");
                printMatrix(lastResult.L.transpose(), "L^T (upper triangular)");
                bool ok = CholeskyDecomposer::verify(*A, lastResult.L);
                std::cout << "Verification L * L^T == A: " << (ok ? "PASS" : "FAIL") << "\n";
                break;
            }
            case 4: {
                if (!lastResult.success) {
                    std::cout << "Compute a decomposition first (option 3).\n";
                    break;
                }
                if (auto b = readVector(lastResult.L.rows())) {
                    auto x = CholeskyDecomposer::solve(lastResult, *b);
                    printVector(x, "x (solution to Ax = b)");
                }
                break;
            }
            case 5: {
                if (!lastResult.success) {
                    std::cout << "Compute a decomposition first (option 3).\n";
                    break;
                }
                std::cout << "det(A) = " << CholeskyDecomposer::determinant(lastResult) << "\n";
                break;
            }
            case 6: {
                A = exampleMatrix();
                lastResult = CholeskyResult{};
                printMatrix(*A, "Loaded example matrix A");
                break;
            }
            case 7:
                std::cout << "Goodbye.\n";
                return 0;
            default:
                std::cout << "Unknown option.\n";
        }
    }
}
