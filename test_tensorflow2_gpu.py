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


def check_gpus():
    print("Tensorflow version %s" % (tf.__version__,))
    print(f"TF git version: {tf.version.GIT_VERSION}")

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


def test_matrix_matrix_dot():
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


def test_batch_matrix_bug():
    """
    This was a bug that was appearing on particular machines with particular
    CUDA versions. Using the same (as far as we could tell) CUDA/driver versions on
    different machines with different GPUs did not produce the bug, as did different
    CUDA versions on the same machine.

    Problem configuration:
      NVIDIA GeForce RTX 3060
      CUDA 11.8
      CUDNN 8.6
    """
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, size=(1, 20, 2)).astype(np.float32)
    c = rng.uniform(-1, 1, size=(2, 2)).astype(np.float32)

    y = x @ c

    x2 = np.tile(x, (2, 1, 1))
    y2 = x2 @ c

    for y2i in y2:
        np.testing.assert_allclose(y2i, y.squeeze(0))

    tols = dict(atol=1e-7, rtol=1e-5)

    z = tf.matmul(x, c).numpy()
    # z = tf.einsum("...tq,...qr->...tr", x, c)
    np.testing.assert_allclose(z, y, **tols)

    z2 = tf.matmul(x2, c).numpy()
    # z2 = tf.einsum("...tq,...qr->...tr", x2, c)
    np.testing.assert_allclose(z2, y2, **tols)


if __name__ == "__main__":
    check_gpus()
    test_matrix_matrix_dot()
    test_batch_matrix_bug()
