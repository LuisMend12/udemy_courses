# Section 4 — Bayesian Optimization

One folder per note, matching the layout used in `rl_course/dynamic_programming/notes`
and `Deep_learning_cnns/notes`. Every `.tex` here is self-contained (IEEEtran, no
shared preamble, no images), so each builds on its own:

```bash
cd gd && pdflatex gaussian_distribution.tex
```

## The section note

| Folder | Note | What it covers |
| --- | --- | --- |
| `bo/` | `bayesian_optimization` | SMBO loop, GP surrogate, EI/PI/LCB, `scikit-optimize`, the GBM worked example |

## The supporting math

These were written to fill gaps that `bo/` leans on without defining. Read in this
order if reading them all; each one references the ones above it.

| Folder | Note | What it covers |
| --- | --- | --- |
| `gd/` | `gaussian_distribution` | Univariate and multivariate density, conjugacy, why Gaussian noise implies least squares |
| `cm/` | `covariance_matrix` | Σ as an object: PSD, spectral geometry, precision matrix, estimation in high dimensions |
| `mg/` | `multivariate_gaussian` | The structure theorems with proofs — affine closure, marginals, conditioning, independence |
| `kr/` | `kernels` | PSD requirement, RKHS, kernel algebra, Matérn/RBF, ARD, kernels for mixed search spaces |

## How they connect

`bo/` is the destination; the other four are the machinery under it.

- The GP posterior equations in `bo/` are the conditioning theorem (`mg/`, Thm 6)
  applied to the joint over observed scores and one new point.
- The claim in `bo/` that posterior variance depends on *where* you sampled and not
  on *what* you observed is `mg/` Corollary 6.1.
- "A Gaussian prior with a Gaussian likelihood is conjugate" is derived in `gd/`.
- The kernel matrix `K` is a covariance matrix (`cm/`), which is why kernels must be
  PSD (`kr/`) and why `K + σ²I` is simultaneously the noise model, the numerical
  jitter, and ridge regularization.
- The Φ and φ in the Expected Improvement formula are defined in `gd/`.
