# Cholesky Decomposition Simulator

An interactive C++ console tool for exploring the Cholesky decomposition
`A = L * L^T` of a symmetric positive-definite matrix `A`. Both the
computation layer and the interactive UI are written in C++, split into a
`backend` (pure numerics, no I/O) and a `frontend` (console menu that reads
input and prints results).

## Project layout

```
cholesky_simulator/
├── backend/
│   ├── Matrix.hpp / Matrix.cpp        # dense matrix: indexing, transpose, multiply, symmetry check
│   └── Cholesky.hpp / Cholesky.cpp    # decompose(), determinant(), solve(), verify()
├── frontend/
│   └── main.cpp                       # menu-driven console UI, calls into backend/
├── Makefile
├── build.ps1
└── README.md
```

The frontend never implements numerical logic itself — it only formats
input/output and calls `CholeskyDecomposer` from the backend. The backend has
no `iostream`/console code, so it can be reused or unit tested independently
of the UI.

## Building

Requires a C++17 compiler (e.g. g++ from MinGW-w64).

**PowerShell:**
```powershell
.\build.ps1
```

**make** (if you have `make` or `mingw32-make` on PATH):
```
make
```

**Manual:**
```
g++ -std=c++17 -Wall -Wextra -O2 -o cholesky_simulator.exe backend/Matrix.cpp backend/Cholesky.cpp frontend/main.cpp
```

## Running

```
.\cholesky_simulator.exe
```

You'll get a menu:

```
=== Cholesky Decomposition Simulator ===
1) Enter a new matrix
2) Check symmetric positive-definite
3) Compute Cholesky decomposition (L, L^T)
4) Solve Ax = b using the decomposition
5) Compute determinant via decomposition
6) Load example matrix
7) Exit
```

1. **Enter a new matrix** — type the size `n`, then the `n x n` entries row
   by row.
2. **Check symmetric positive-definite** — reports whether the loaded matrix
   qualifies for Cholesky decomposition, without discarding it.
3. **Compute Cholesky decomposition** — prints `L` and `L^T`, then verifies
   `L * L^T == A` and reports PASS/FAIL.
4. **Solve Ax = b** — after a successful decomposition (option 3), enter the
   right-hand side vector `b`; solves via forward substitution (`L y = b`)
   then back substitution (`L^T x = y`).
5. **Compute determinant** — `det(A) = product(L_ii)^2`, using the existing
   decomposition.
6. **Load example matrix** — loads the classic textbook example
   `[[4,12,-16],[12,37,-43],[-16,-43,98]]` (its known Cholesky factor is
   `[[2,0,0],[6,1,0],[-8,5,3]]`, determinant `36`).
7. **Exit**.

## Math background

For symmetric positive-definite `A`, the Cholesky-Banachiewicz algorithm
computes lower-triangular `L` column by column:

- Diagonal: `L(j,j) = sqrt(A(j,j) - sum_{k<j} L(j,k)^2)`
- Off-diagonal: `L(i,j) = (A(i,j) - sum_{k<j} L(i,k) L(j,k)) / L(j,j)`

If any diagonal term under the square root is non-positive, `A` is not
positive definite and the decomposition fails — the backend reports this
instead of computing `NaN` results.

## Notes

- `hyperparameter_optimization_for_ml/gaussian_basics/cholesky_decomposition.cpp`
  and `.py` (one directory up) are the original single-file integer-matrix
  references this simulator was built from; this project generalizes them to
  `double`, adds validation, solving, determinant, and the interactive UI.
