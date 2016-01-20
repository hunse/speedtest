from __future__ import print_function

import os
import timeit

import numpy as np

gpuflags = 'device=gpu,floatX=float32'
if os.environ.has_key('THEANO_FLAGS'):
    os.environ['THEANO_FLAGS'] += gpuflags
else:
    os.environ['THEANO_FLAGS'] = gpuflags
import theano
import theano.tensor as T
dtype = theano.config.floatX

print("Theano location: %s" % theano.__file__)
print("Theano version: %s" % theano.__version__)

# --- create Theano function
sA = T.matrix(dtype=dtype)
sB = T.matrix(dtype=dtype)
sC = T.dot(sA,sB)
f = theano.function([sA, sB], sC)

# --- create test data
m = 10000
k = 2000
n = 1500

rng = np.random.RandomState(0)

A = np.asarray(rng.randn(m, k), dtype=dtype)
B = np.asarray(rng.randn(k, n), dtype=dtype)

# --- run tests
N = 10  # number of repeats, for tests

t = min(timeit.Timer('np.dot(A,B)', 'from __main__ import np, A, B'
                 ).repeat(N, 1))
print("Numpy: multiplied (%d,%d) matrix with (%d,%d) matrix in %0.3f s"
      % (m,k,k,n,t))

t = min(timeit.Timer('f(A,B)', 'from __main__ import f, A, B'
                 ).repeat(N, 1))
print("Theano (GPU): multiplied (%d,%d) matrix with (%d,%d) matrix in %0.3f s"
      % (m,k,k,n,t))
