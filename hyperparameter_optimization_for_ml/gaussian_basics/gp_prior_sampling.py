# Sampling from P(f): draw functions from a Gaussian process prior
# using the squared-exponential (RBF) kernel and a Cholesky factorization.

from __future__ import division
import numpy as np
import matplotlib.pyplot as pl


def kernel(a, b):
    """ GP squared exponential kernel """
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-.5 * sqdist)


n = 50                                        # number of test points.
Xtest = np.linspace(-5, 5, n).reshape(-1, 1)  # Test points.
K_ = kernel(Xtest, Xtest)                     # Kernel at test points.

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6 * np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n, 10)))

pl.plot(Xtest, f_prior)
pl.title('10 samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.show()
