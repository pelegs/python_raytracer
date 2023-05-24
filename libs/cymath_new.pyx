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

cdef int sign_int(int n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0

cdef double sign_double(double x):
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    else:
        return 0.0

def sign(x):
    if isinstance(x, int):
        return sign_int(x)
    elif isinstance(x, float):
        return sign_double(x)
    else:
        raise ValueError(f"x must be int or double, not {type(x)}!")


#####################
# Vector operations #
#####################

# dot product
cdef double dot_cy(
        double [:] u,
        double [:] v,
    ):
    cdef int ul = u.shape[0]
    if ul != v.shape[0]:
        raise ValueError("Vectors must be of identical length.")
    cdef double s = 0.0
    for i in range(ul):
        s += u[i]*v[i]
    return s

def dot(
        double[:] u not None,
        double[:] v not None,
    ):
    return dot_cy(u, v)

# cross product
cdef double[:] cross_cy(
        double[:] u,
        double[:] v,
    ):
    cdef double[:] w = np.array(
        [
            u[1]*v[2] - u[2]*v[1],
            u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0],
        ]#, dtype=np.double
    )
    return w

def cross(
        double[:] u not None,
        double[:] v not None,
    ):
    return np.asarray(cross_cy(u, v))


# Norm2 and norm
cdef double norm2_cy(
        double[:] v
    ):
    return dot_cy(v, v)

cdef double norm_cy(
        double[:] v
    ):
    return sqrt(dot_cy(v, v))

def norm2(
        double[:] v not None,
    ):
    return norm2_cy(v)

def norm(
        double[:] v not None,
    ):
    return norm_cy(v)


# Unit (normalized) vecs
cdef double[:] unit_cy(
        double[:] v
    ):
    # if not np.any(v):
    #     raise ValueError("Can't normalize the zero vector!")
    cdef int v_dim = v.shape[0]
    cdef double v_norm = norm_cy(v)
    cdef double[:] v_unit = np.zeros(v_dim)
    for i in range(v_dim):
        v_unit[i] = v[i]/v_norm
    return v_unit


def unit(
        double[:] v not None,
    ):
    return np.asarray(unit_cy(v))
