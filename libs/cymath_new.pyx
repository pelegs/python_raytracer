import numpy as np
cimport numpy as np
from libc.math cimport pi, sqrt, sin, cos, tan, acos, fmin
cimport cython
import cython
#cython: boundscheck=False, wraparound=False, nonecheck=False

ctypedef double DTYPE_t


###################
# Math operations #
###################

cdef double sign(double x):
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    else:
        return 0.0


#####################
# Vector operations #
#####################

# dot product
def dot2d(
        double[:] u not None,
        double[:] v not None,
    ):
    return u[0]*v[0] + u[1]*v[1]

def dot(
        double[:] u not None,
        double[:] v not None,
    ):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def dot_quat(
        double[:] u not None,
        double[:] v not None,
    ):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2] + u[3]*v[3]


# cross product
cdef double[:] cross(
        double[:] u,
        double[:] v,
    ):
    cdef double[:] w = np.zeros(3)
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]
    return w


# Norm2 and norm
def norm2(double[:] v):
    return dot(v, v)

def norm(double[:] v):
    return sqrt(norm2(v))


# Unit (normalized) vecs
cdef double[:] unit_cy(
        double[:] v
    ):
    # if not np.any(v):
    #     raise ValueError("Can't normalize the zero vector!")
    cdef int v_dim = v.shape[0]
    cdef double v_norm = norm(v)
    cdef double[:] v_unit = np.zeros(v_dim)
    for i in range(v_dim):
        v_unit[i] = v[i]/v_norm
    return v_unit


def unit(
        double[:] v not None,
    ):
    return np.asarray(unit_cy(v))
