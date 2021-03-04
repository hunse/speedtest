"""
To install tensorflow for GPU:
    pip uninstall tensorflow
    pip install tensorflow-gpu
"""
import sys
import os
import timeit

# turn off INFO https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf

print("Tensorflow version %s" % (tf.__version__,))

sys_details = tf.sysconfig.get_build_info()
print("CUDA version %s" % (sys_details["cuda_version"],))
print("Sys details: %s" % (sys_details,))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("Found %d GPU(s): %s" % (len(gpus), gpus))
else:
    print("DID NOT FIND ANY GPUs!")
    sys.exit(1)


def test(fn, number):
    timer = timeit.Timer('fn()', 'from __main__ import fn')
    times = timer.repeat(repeat=number, number=1)
    return min(times)


# --- test matrix-matrix dot
rng = np.random.RandomState(3)

n = 5000
A = rng.uniform(-1, 1, size=(n, n))
B = rng.uniform(-1, 1, size=(n, n))
# print(np.dot(A, B))

with tf.device("/GPU:0"):
    a = tf.constant(A, name='a')
    b = tf.constant(B, name='b')

    # TF2 appears to use caching, subsequent runs happen instantaneously. Just run once.
    timer = timeit.default_timer()
    c = tf.matmul(a, b)
    t = timeit.default_timer() - timer

# fn = lambda: tf.matmul(a, b)
# t = test(fn, number=1)

print("multiplied two (%d,%d) matrices in %0.1f ms" % (n, n, 1e3 * t))
