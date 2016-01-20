from __future__ import print_function

import timeit

import numpy as np
import numpy.random as npr

def test(fn, number):
    timer = timeit.Timer('fn()', 'from __main__ import fn')
    return min(timer.repeat(repeat=number, number=1))

print("Numpy location: %s" % np.__file__)
print("Numpy version: %s" % np.__version__)
print("Numpy config:")
np.show_config()

# --- test matrix-matrix dot
n = 1000
A = npr.randn(n,n)
B = npr.randn(n,n)

fn = lambda: np.dot(A, B)
t = test(fn, number=10)
print("multiplied two (%d,%d) matrices in %0.1f ms" % (n, n, 1e3 * t))

# --- test matrix-vector dot
n = 2000
A = npr.randn(n, n)
B = npr.randn(n)

fn = lambda: np.dot(A, B)
t = test(fn, number=100)
print("multiplied (%d,%d) matrix and (%d) vector in %0.1f ms" % (
        n, n, n, 1e3 * t))

# --- test vector-vector dot
n = 4000
A = npr.randn(n)
B = npr.randn(n)

fn = lambda: np.dot(A, B)
t = test(fn, number=1000)
print("dotted two (%d) vectors in %0.2f us" % (n, 1e6*t))

# --- test SVD
m, n = (2000,1000)
A = npr.randn(m, n)

fn = lambda: np.linalg.svd(A, full_matrices=False)
t = test(fn, number=1)
print("SVD of ({:d},{:d}) matrix in {:0.3f} s".format(m, n, t))

# --- test Eigendecomposition
n = 1500
A = npr.randn(n, n)
fn = lambda: np.linalg.eig(A)
t = test(fn, number=1)
print("Eigendecomposition of ({:d},{:d}) matrix in {:0.3f} s".format(n, n, t))

# --- My results

# --- New lab computer, OpenBLAS from source
# multiplied two (1000,1000) matrices in 13.1 ms
# dotted two (4000) vectors in 0.95 us
# SVD of (2000,1000) matrix in 0.504 s
# Eigendecomposition of (1500,1500) matrix in 3.671 s
