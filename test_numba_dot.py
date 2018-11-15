import time

import numpy as np
import numba
from numba import njit

def test(funcs, A, B, trials):
    print("Test: A %s, B %s, trials=%d" % (A.shape, B.shape, trials))
    for func in funcs:
        t = time.time()
        for i in range(trials):
            func(A, B)
        td = time.time() - t
        print(func.__name__, td)

def np_dot(A, B):
    return np.dot(A, B)

@njit
def numba_dot(A, B):
    return np.dot(A, B)


print("Numpy location: %s" % np.__file__)
print("Numpy version: %s" % np.__version__)
print("Numpy config:")
np.show_config()

print("Numba location: %s" % numba.__file__)
print("Numba version: %s" % numba.__version__)

funcs = (np_dot, numba_dot)
trials = 500
n = 1000
d = 10
m = 5000

A = np.random.randn(n, d)
B = np.random.randn(d, m)
test(funcs, A, B, trials=trials)

trials = 10
n = 3000
d = 3000
m = 5000

A = np.random.randn(n, d)
B = np.random.randn(d, m)
test(funcs, A, B, trials=trials)
