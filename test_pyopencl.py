"""
Basic tests for maximum flops and bandwidth of card.
"""
import sys
# import timeit
import time

import numpy as np

import mako.template
import pyopencl as cl
import pyopencl.array

PY2 = sys.version_info[0] == 2


def as_ascii(string):
    if not PY2 and isinstance(string, bytes):  # Python 3
        return string.decode('ascii')
    elif not isinstance(string, str):
        return str(string)
    else:
        return string


def round_up(x, n):
    return int(np.ceil(float(x) / n)) * n


def time_kernel(kernel, queue, *args, **kwargs):
    t = time.time()
    kernel(queue, *args, **kwargs)
    queue.finish()
    t = time.time() - t
    return t


print("PyOpenCL location: %s" % cl.__file__)
print("PyOpenCL version: %s" % cl.version.VERSION_TEXT)

# --- setup
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mwgsize = queue.device.max_work_group_size

rng = np.random.RandomState(9)

# n = int(1e6)
# n = int(1e7)
n = int(1e8)

# X = rng.uniform(-1, 1, size=n).astype(np.float32)
X = np.arange(n, dtype=np.float32)
clX = cl.array.Array(queue, X.shape, X.dtype)
clY = cl.array.Array(queue, X.shape, X.dtype)
clX.set(X)

lsize = None
gsize = (n,)

# --- flops
# ops_per_item = 2
# ops_per_item = 128
# ops_per_item = 1024
ops_per_item = 4096
flops_per_call = ops_per_item * n
bw_per_call = clX.nbytes + clY.nbytes

source = """
__kernel void fn(
    __global const float *x,
    __global float *y)
{
    const int i = get_global_id(0);

    float value = x[i];
    for (int k = 0; k < ${ops_per_item}; k++) {
        value *= 1.001f;
    }
    y[i] = value;
}
"""

textconf = dict(ops_per_item=ops_per_item)
source = as_ascii(mako.template.Template(
    source, output_encoding='ascii').render(**textconf))
kernel = cl.Program(ctx, source).build().fn

t = time_kernel(kernel, queue, gsize, lsize, clX.data, clY.data)
print("Mults: %0.2f ms" % (1000 * t))
print("  GF/s: %0.2f" % (1e-9 * flops_per_call / t))
print("  GB/s: %0.2f" % (1e-9 * bw_per_call / t))


# --- bandwidth
source = """
__kernel void fn(
    __global const float *x,
    __global float *y)
{
    const int i = get_global_id(0);
    y[i] = x[i];
}
"""

textconf = dict()
source = as_ascii(mako.template.Template(
    source, output_encoding='ascii').render(**textconf))
kernel = cl.Program(ctx, source).build().fn

t = time_kernel(kernel, queue, gsize, lsize, clX.data, clY.data)
print("Copy: %0.2f ms" % (1000 * t))
# print("  GF/s: %0.2f" % (1e-9 * flops_per_call / t))
print("  GB/s: %0.2f" % (1e-9 * bw_per_call / t))


# --- read bandwidth
n2 = int(1e6)
m2 = 32
# m2 = 64
# n2 = int(1e6)
# m2 = 1

X2 = np.arange(n2, dtype=np.float32)
clX2 = cl.array.Array(queue, X2.shape, X2.dtype)
clY2 = cl.array.Array(queue, X2.shape, X2.dtype)

X2tall = np.arange(n2*m2, dtype=np.float32).reshape(n2, m2)
clX2tall = cl.array.Array(queue, X2tall.shape, X2tall.dtype)
clY2tall = cl.array.Array(queue, X2tall.shape, X2tall.dtype)
clX2tall.set(X2tall)

bw_per_call = clX2tall.nbytes + clY2.nbytes

source = """
__kernel void fn(
    __global const float *x,
    __global float *y)
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);

    const int tj = get_local_id(0);
    const int ti = get_local_id(1);

    __local float xlocal[${lsizei}][${lsizej}];
    xlocal[ti][tj] = (i < ${n2}) ? x[i*${m2} + j] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

% for k in range(int(np.log2(m2))-1, 0, -1):
    if (tj < ${2**k})  xlocal[ti][tj] += xlocal[ti][tj + ${2**k}];
    barrier(CLK_LOCAL_MEM_FENCE);
% endfor
    if (tj == 0)
        y[i] = xlocal[ti][0] + xlocal[ti][1];
}
"""

lsize = (m2, mwgsize/m2)
gsize = (m2, round_up(n2, lsize[1]))

textconf = dict(np=np, m2=m2, n2=n2, lsizej=lsize[0], lsizei=lsize[1])
source = as_ascii(mako.template.Template(
    source, output_encoding='ascii').render(**textconf))
kernel = cl.Program(ctx, source).build().fn

t = time_kernel(kernel, queue, gsize, lsize, clX2tall.data, clY2.data)
print("Sum local: %0.2f ms" % (1000 * t))
print("  GB/s: %0.2f" % (1e-9 * bw_per_call / t))


# --- read bandwidth 2
X2wide = np.arange(n2*m2, dtype=np.float32).reshape(m2, n2)
clX2wide = cl.array.Array(queue, X2wide.shape, X2wide.dtype)
clY2wide = cl.array.Array(queue, X2wide.shape, X2wide.dtype)
clX2wide.set(X2wide)

bw_per_call = clX2wide.nbytes + clY2.nbytes

source = """
__kernel void fn(
    __global const float *x,
    __global float *y)
{
    const int i = get_global_id(0);

    float sum = 0;
    for (int j = 0; j < ${m2}; j++)
        sum += x[j*${n2} + i];

    y[i] = sum;
}
"""

lsize = None
gsize = (n2,)

textconf = dict(m2=m2, n2=n2)
source = as_ascii(mako.template.Template(
    source, output_encoding='ascii').render(**textconf))
kernel = cl.Program(ctx, source).build().fn

t = time_kernel(kernel, queue, gsize, lsize, clX2wide.data, clY2.data)
print("Sum wide: %0.2f ms" % (1000 * t))
print("  GB/s: %0.2f" % (1e-9 * bw_per_call / t))


# --- read bandwidth 3
bw_per_call = clX2tall.nbytes + clY2.nbytes

source = """
__kernel void fn(
    __global const float *x,
    __global float *y)
{
    const int i = get_global_id(0);

    float sum = 0;
    for (int j = 0; j < ${m2}; j++)
        sum += x[i*${m2} + j];

    y[i] = sum;
}
"""

lsize = None
gsize = (n2,)

textconf = dict(m2=m2, n2=n2)
source = as_ascii(mako.template.Template(
    source, output_encoding='ascii').render(**textconf))
kernel = cl.Program(ctx, source).build().fn

t = time_kernel(kernel, queue, gsize, lsize, clX2tall.data, clY2.data)
print("Sum tall: %0.2f ms" % (1000 * t))
print("  GB/s: %0.2f" % (1e-9 * bw_per_call / t))


# --- write bandwidth
bw_per_call = clX2.nbytes + clY2tall.nbytes

source = """
__kernel void fn(
    __global const float *x,
    __global float *y)
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);
    if (i >= ${n2})
        return;

    __local float xi;
    if (j == 0)
        xi = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    y[i*${m2} + j] = xi;
}
"""

lsize = (m2, 4)
gsize = (m2, round_up(n2, lsize[1]))

textconf = dict(m2=m2, n2=n2)
source = as_ascii(mako.template.Template(
    source, output_encoding='ascii').render(**textconf))
kernel = cl.Program(ctx, source).build().fn

t = time_kernel(kernel, queue, gsize, lsize, clX2.data, clY2tall.data)
print("multi-write local: %0.2f ms" % (1000 * t))
print("  GB/s: %0.2f" % (1e-9 * bw_per_call / t))


# --- write bandwidth 2
bw_per_call = clX2.nbytes + clY2wide.nbytes

source = """
__kernel void fn(
    __global const float *x,
    __global float *y)
{
    const int i = get_global_id(0);

    const float xi = x[i];
    for (int j = 0; j < ${m2}; j++)
        y[j*${n2} + i] = xi;
}
"""

lsize = None
gsize = (n2,)

textconf = dict(m2=m2, n2=n2)
source = as_ascii(mako.template.Template(
    source, output_encoding='ascii').render(**textconf))
kernel = cl.Program(ctx, source).build().fn

t = time_kernel(kernel, queue, gsize, lsize, clX2.data, clY2wide.data)
print("multi-write wide: %0.2f ms" % (1000 * t))
print("  GB/s: %0.2f" % (1e-9 * bw_per_call / t))


# --- write bandwidth 3
bw_per_call = clX2.nbytes + clY2tall.nbytes

source = """
__kernel void fn(
    __global const float *x,
    __global float *y)
{
    const int i = get_global_id(0);

    const float xi = x[i];
    for (int j = 0; j < ${m2}; j++)
        y[i*${m2} + j] = xi;
}
"""

lsize = None
gsize = (n2,)

textconf = dict(m2=m2, n2=n2)
source = as_ascii(mako.template.Template(
    source, output_encoding='ascii').render(**textconf))
kernel = cl.Program(ctx, source).build().fn

t = time_kernel(kernel, queue, gsize, lsize, clX2.data, clY2tall.data)
print("multi-write tall: %0.2f ms" % (1000 * t))
print("  GB/s: %0.2f" % (1e-9 * bw_per_call / t))
