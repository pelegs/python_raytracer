import numpy as np
from scipy import Rotation


PRECISION = 1E-8
X_DIR = np.array([1, 0, 0])
Y_DIR = np.array([0, 1, 0])
Z_DIR = np.array([0, 0, 1])

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


def rotate_ves(vecs, rot):
    r = Rotation.from_rotvec(rot)
    return r.apply(vecs)


if __name__ == "__main__":
    pass
