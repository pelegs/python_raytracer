import numpy as np
from libs.cymath import *


X_, Y_, Z_ = np.identity(3)

v = np.random.uniform(-10, 10, size=3)
v2 = scale_to(v, 10)
print(norm(v2))
print(same_direction(v, v2, 1.5E-7))
