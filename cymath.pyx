import numpy as np
cimport numpy as np
cimport cython
#cython: boundscheck=False, wraparound=False, nonecheck=False

ctypedef double DTYPE_t


def dot(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v
        ):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def cy_cross(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v
        ):
    cdef np.ndarray[DTYPE_t, ndim=1] w = np.zeros(3)
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]
    return w

def line_plane_intersection(
        np.ndarray[DTYPE_t, ndim=1] line_start,
        np.ndarray[DTYPE_t, ndim=1] line_dir,
        np.ndarray[DTYPE_t, ndim=1] plane_normal_form,
        ):
    cdef double D = dot(line_dir, plane_normal_form[:3])
    if D == 0.0:
        return None
    cdef double A = dot(line_start, plane_normal_form[:3]) + plane_normal_form[3]
    return -A/D

def point_in_triangle(
        np.ndarray[DTYPE_t, ndim=1] p,
        np.ndarray[DTYPE_t, ndim=2] vertices,
        ):
    cdef np.ndarray[DTYPE_t, ndim=2] tran_vertices = vertices - p
    u = cy_cross(tran_vertices[1], tran_vertices[2])
    v = cy_cross(tran_vertices[2], tran_vertices[1])
    w = cy_cross(tran_vertices[0], tran_vertices[1])
    if dot(u, v) < 0.0:
        return False
    if dot(u, w) < 0.0:
        return False
    return True

def reflect(
        np.ndarray[DTYPE_t, ndim=1] r,
        np.ndarray[DTYPE_t, ndim=1] n,
        ):
    return r - 2*(dot(r, n))*n
