"""
To install tensorflow for GPU:
    pip uninstall tensorflow
    pip install tensorflow-gpu
"""

from __future__ import print_function

import timeit

import numpy as np
import tensorflow as tf

rng = np.random


def test(fn, number):
    timer = timeit.Timer('fn()', 'from __main__ import fn')
    return min(timer.repeat(repeat=number, number=1))


# --- test matrix-matrix dot
n = 5000
A = rng.uniform(-1, 1, size=(n, n))
B = rng.uniform(-1, 1, size=(n, n))

with tf.device('/gpu:0'):
    a = tf.constant(A, name='a')
    b = tf.constant(B, name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    fn = lambda: sess.run(c)
    t = test(fn, number=3)
    print("multiplied two (%d,%d) matrices in %0.1f ms" % (n, n, 1e3 * t))
