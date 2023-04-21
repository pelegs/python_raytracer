import numpy as np


PRECISION = 1E-8


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


if __name__ == "__main__":
    pass
