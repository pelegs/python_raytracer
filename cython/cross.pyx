import numpy as np
cimport numpy as np
cimport cython
#cython: boundscheck=False, wraparound=False, nonecheck=False

ctypedef double DTYPE_t

def cross(np.ndarray[DTYPE_t, ndim=1] u, np.ndarray[DTYPE_t, ndim=1] v):
    cdef np.ndarray[DTYPE_t, ndim=1] w = np.zeros(3)
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]
    return w
