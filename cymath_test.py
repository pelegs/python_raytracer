import numpy as np
from libs.cymath import *


X_, Y_, Z_ = np.identity(3)

v = np.array([1, 1, 0], dtype=np.double)
q = get_rotation(v, X_)
print(q)
print(np.sin(np.pi/8), np.cos(np.pi/8))
