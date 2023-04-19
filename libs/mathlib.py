import numpy as np


def unit(v):
    L = np.linalg.norm(v)
    if L == 0:
        raise ValueError("Can't normalize zero vectors")
    return v/L


if __name__ == "__main__":
    pass
