import sys
import os
import timeit

# turn off INFO https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("Pytorch version %s" % (torch.__version__,))
print("CUDA version %s" % (torch.version.cuda,))
print("CUDA home is %s" % (CUDA_HOME,))
print("CUDNN version %s" % (torch.backends.cudnn.version(),))
print("CUDNN is ENABLED" if torch.backends.cudnn.enabled else "CUDNN is NOT enabled")

gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
if gpus:
    print("Found %d GPU(s): %s" % (len(gpus), gpus))
    assert torch.cuda.is_available(), "CUDA not available!"
else:
    print("DID NOT FIND ANY GPUs!")
    sys.exit(1)

device = torch.device("cuda:0")
torch.cuda.init()
assert torch.cuda.is_initialized(), "CUDA not initialized!"

# --- test matrix-matrix dot
rng = np.random.RandomState(3)

n = 5000
A = rng.uniform(-1, 1, size=(n, n))
B = rng.uniform(-1, 1, size=(n, n))

a = torch.from_numpy(A).to(device)
b = torch.from_numpy(B).to(device)
torch.cuda.synchronize()

# this method of timing seems to get a reasonable time (matching TensorFlow)
# - Adding a `synchronize` in the loop makes things much too slow (not sure why)
# - For torch 1.8.2, the time is not negligible. Newer torch versions may have lazier
#   evaluation, though, which could make it negligible.
timer = timeit.default_timer()
c = torch.matmul(a, b)
t = timeit.default_timer() - timer

print("multiplied two (%d,%d) matrices in %0.1f ms" % (n, n, 1e3 * t))
