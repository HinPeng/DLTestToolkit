from __future__ import print_function

_PREFIX = 'workspace'  # the prefix dir name relative to $HOME

_TOTAL_MEM = 16  # GB
_FRAC_TO_SUBTRACT = 0.0
_LOGNODE_TIME = False
_ALLOW_GROWTH = False
_SINGLE_GPU_ID = 1
_USE_CUDA_MPS = False

# SharedSession config
_ALLOW_SHARE = True
_TARGET = "grpc://localhost:29999"
_TF_USE_ALIGNED_SCHEDULING = False