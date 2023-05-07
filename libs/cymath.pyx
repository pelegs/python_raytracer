import numpy as np
cimport numpy as np
from libc.math cimport pi, sqrt, sin, cos, acos
cimport cython
#cython: boundscheck=False, wraparound=False, nonecheck=False

ctypedef double DTYPE_t


#############
# Constants #
#############

PRECISION = 1E-8
I_ = np.identity(3, dtype=np.double)
X_, Y_, Z_ = I_
O_ = np.zeros(3, dtype=np.double)
HALVES_2D = np.array([0.5, 0.5], dtype=np.double)
HALVES_2D_Z0 = np.array([0.5, 0.5, 0.0], dtype=np.double)
CORNERS_FROM_CENTER = np.array([
    [-0.5, -0.5],  # NW
    [0.5, -0.5],   # NE
    [0.5, 0.5],    # SE
    [-0.5, 0.5],   # SW
])

# Screen resolutions
VGA_480p_4_3 = (640, 480)
VGA_480p_3_2 = (720, 480)
VGA_480p_2_1 = (960, 480)
VGA_480p_16_9 = (848, 480)
HD_720p_4_3 = (960, 720)
HD_720p_16_9 = (1280, 720)
HD_1080p_16_9 = (1920, 1080)


#####################
# Vector operations #
#####################

def dot(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v
        ):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def cross(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v
        ):
    cdef np.ndarray[DTYPE_t, ndim=1] w = np.zeros(3)
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]
    return w

def norm2(np.ndarray[DTYPE_t, ndim=1] v):
    return dot(v, v)

def norm(np.ndarray[DTYPE_t, ndim=1] v):
    return sqrt(norm2(v))

def unit(np.ndarray[DTYPE_t, ndim=1] v):
    cdef double L = norm2(v)
    if L == 0:
        raise ValueError("Can't normalize zero vectors")
    return v/sqrt(L)

def scale_to(
        np.ndarray[DTYPE_t, ndim=1] vec,
        double L,
    ):
    return L*unit(vec)

def same_direction(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v,
        double precision,
    ):
    return np.allclose(unit(u), unit(v), atol=precision)

def distance2(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v
    ):
    return norm2(u-v)

def distance(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v
    ):
    return norm(u-v)

def angle_between(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v,
    ):
    cdef double c = dot(u, v)/(norm(u)*norm(v))
    if -1.0 <= c <= 1.0:
        return acos(c)
    raise ValueError(f"{c} is not a valid argument to acos.")


###############
# Quaternions #
###############

def quaternion(
        np.ndarray[DTYPE_t, ndim=1] ax,
        double t,
    ):
    cdef double s = sin(t/2)
    cdef double c = cos(t/2)
    cdef np.ndarray[DTYPE_t, ndim=1] q = np.array([s*ax[0], s*ax[1], s*ax[2], c])
    return q

def qprod_v(
        np.ndarray[DTYPE_t, ndim=1] q,
        np.ndarray[DTYPE_t, ndim=1] v,
    ):
    cdef np.ndarray[DTYPE_t, ndim=1] t = 2*cross(q[:3], v)
    return v + q[3]*t + cross(q[:3], t)

def qprod_M(
        np.ndarray[DTYPE_t, ndim=1] q,
        np.ndarray[DTYPE_t, ndim=2] M,
        ):
    cdef int n = M.shape[0]
    cdef int m = M.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] u = np.zeros((n, m))
    cdef np.ndarray[DTYPE_t, ndim=1] t = np.zeros(q.shape[0])
    for i in range(M.shape[0]):
        t = 2*cross(q[:3], M[i])
        u[i] = M[i] + q[3]*t + cross(q[:3], t)
    return u

def rotate_v(
        np.ndarray[DTYPE_t, ndim=1] ax,
        np.ndarray[DTYPE_t, ndim=1] vec,
        double t,
    ):
    if t == 0:
        return vec
    elif t == pi:
        return -1.0*vec
    cdef np.ndarray[DTYPE_t, ndim=1] q = quaternion(ax, t)
    return qprod_v(q, vec)

def rotate_M(
        np.ndarray[DTYPE_t, ndim=2] vecs,
        np.ndarray[DTYPE_t, ndim=1] ax,
        double t,
    ):
    if t == 0:
        return vecs
    elif t == pi:
        return -1.0*vecs
    cdef np.ndarray[DTYPE_t, ndim=1] q = quaternion(ax, t)
    return qprod_M(q, vecs)

def rotate_v_by_q(
        np.ndarray[DTYPE_t, ndim=1] vec,
        np.ndarray[DTYPE_t, ndim=1] q,
    ):
    return qprod_v(q, vec)

def rotate_M_by_q(
        np.ndarray[DTYPE_t, ndim=2] vecs,
        np.ndarray[DTYPE_t, ndim=1] q,
    ):
    return qprod_M(q, vecs)

def get_rotation(
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v,
    ):
    cdef double t = angle_between(u, v)
    cdef np.ndarray[DTYPE_t, ndim=1] r = unit(cross(u, v))
    cdef np.ndarray[DTYPE_t, ndim=1] q = quaternion(r, t)
    return q

def rotate_x(
        np.ndarray[DTYPE_t, ndim=2] vecs,
        double t,
    ):
    cdef np.ndarray[DTYPE_t, ndim=1] ax = np.array([1.0, 0.0, 0.0])
    return rotate_M(vecs, ax, t)

def rotate_y(
        np.ndarray[DTYPE_t, ndim=2] vecs,
        double t,
    ):
    cdef np.ndarray[DTYPE_t, ndim=1] ax = np.array([0.0, 1.0, 0.0])
    return rotate_M(vecs, ax, t)

def rotate_z(
        np.ndarray[DTYPE_t, ndim=2] vecs,
        double t,
    ):
    cdef np.ndarray[DTYPE_t, ndim=1] ax = np.array([0.0, 0.0, 1.0])
    return rotate_M(vecs, ax, t)

def rotate_around_point(
        np.ndarray[DTYPE_t, ndim=2] vecs,
        np.ndarray[DTYPE_t, ndim=1] ax,
        np.ndarray[DTYPE_t, ndim=1] point,
        double t,
    ):
    cdef np.ndarray[DTYPE_t, ndim=2] vecs_trans = vecs - point
    vecs_trans = rotate_M(vecs_trans, ax, t)
    return vecs_trans + point

def rotate_to(
        np.ndarray[DTYPE_t, ndim=2] vecs,
        np.ndarray[DTYPE_t, ndim=1] u,
        np.ndarray[DTYPE_t, ndim=1] v,
    ):
    cdef np.ndarray[DTYPE_t, ndim=1] ax = get_rotation(u, v)[:3]
    cdef double t = angle_between(u, v)
    return rotate_M(vecs, ax, t)


###############
# Other stuff #
###############

def point_on_plane(
        np.ndarray[DTYPE_t, ndim=1] plane_normal,
        np.ndarray[DTYPE_t, ndim=1] point,
        double pr,
        ):
    cdef double x = dot(plane_normal[:3], point)
    cdef double y = -plane_normal[3]
    return y-pr <= x <= y+pr

def line_at_point(
        np.ndarray[DTYPE_t, ndim=1] line_start,
        np.ndarray[DTYPE_t, ndim=1] line_dir,
        double t,
        ):
    return line_start + t*line_dir

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
        np.ndarray[DTYPE_t, ndim=2] vertices,
        np.ndarray[DTYPE_t, ndim=1] p,
        ):
    cdef np.ndarray[DTYPE_t, ndim=2] tran_vertices = vertices - p
    u = cross(tran_vertices[1], tran_vertices[2])
    v = cross(tran_vertices[2], tran_vertices[0])
    w = cross(tran_vertices[0], tran_vertices[1])
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
