import numpy as np
from scipy.spatial.transform import Rotation


# General constants
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


def unit(v):
    L = np.linalg.norm(v)
    if L == 0:
        raise ValueError("Can't normalize zero vectors")
    return v/L


def distance(v1, v2):
    return np.linalg.norm(v2-v1)


def distance2(v1, v2):
    dv = v2-v1
    return np.dot(dv, dv)


def norm2(v):
    return np.dot(v, v)


def angle_between(v1, v2, units=False):
    c = np.dot(v1, v2)
    if not units:
        c = c / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if -1 <= c <= 1:
        return np.arccos(c)
    raise ValueError(f"{c} is not a valid argument to acos.")


def py_cross(v1, v2):
    """ The cross product needs to be redefined because of an annoying """
    """ behaviour of pyright: it marks anything following cross() as """
    """ unreachable (similar issue with pylance: """
    """ https://github.com/microsoft/pylance-release/issues/3195) """
    """ It is what it is ¯\_(ツ)_/¯ """
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2
    return np.array([
        v1y*v2z - v1z*v2y,
        v1z*v2x - v1x*v2z,
        v1x*v2y - v1y*v2x
    ])


def rotation_x(t):
    s, c = np.sin(t/2), np.cos(t/2)
    return np.array([s, 0, 0, c])


def rotation_y(t):
    s, c = np.sin(t/2), np.cos(t/2)
    return np.array([0, s, 0, c])


def rotation_z(t):
    s, c = np.sin(t/2), np.cos(t/2)
    return np.array([0, 0, s, c])


def get_rotation(vs, vt):
    """
    Returns quaternion q which rotates vs s.t. it is pointing in the
    direction of vt.
    TODO: make it work for vectors 180° apart.
    """
    t = angle_between(vs, vt)
    s, c = np.sin(t/2), np.cos(t/2)
    r = unit(py_cross(vs, vt))
    q = np.append(r*s, c)
    return q


def rotate_vecs(vecs, q):
    """
    Applies the rotation given by the quaternion q to the set ofnvectors vecs.
    """
    return Rotation.from_quat(q).apply(vecs)


def rotate_to(vs, vt, vecs):
    """
    Rotates the set of vectors vecs by the same quaternion which
    rotates vs to vt.
    """
    q = get_rotation(vs, vt)
    return rotate_vecs(vecs, q)


def rotate_around(vecs, q, center):
    return rotate_vecs(vecs-center, q) + center


def same_direction(v1, v2, precision=PRECISION):
    """
    Checks whether the two vectors v1 and v2 are similar to within a precision.
    """
    return np.allclose(unit(v1), unit(v2), atol=precision)


def scale_to(vec, length):
    return unit(vec) * length


if __name__ == "__main__":
    pass
