import numpy as np
from libs.mathlib import *
from libs.classes import *

""" from scipy.spatial.transform import Rotation as Rot """


np.set_printoptions(precision=3, suppress=True)

# Testing angles are correct
""" v_in = I_ """
""" dt = np.pi/2 """
""" for t in np.arange(0, 2*np.pi+dt/2, dt): """
"""     c = np.cos(t/2) """
"""     s = np.sin(t/2) """
"""     q = np.append(Z_*s, c) """
"""     R = Rot.from_quat(q) """
"""     v_out = R.apply(v_in) """
"""     print(f"θ={t:0.3f} ({np.degrees(t):0.3f}°), cos(θ)={c:0.3f}") """
"""     print(f"q={q}") """
"""     print(f"v_in=\n{v_in}") """
"""     print(f"v_out=\n{v_out}") """
"""     print("-----------------------") """


# Testing rotation is correct
""" for _ in range(10000): """
"""     vs, vt = np.random.uniform(-1, 1, (2, 3)) """
"""     r = get_rotation(vs, vt) """
"""     vout = rotate_vecs(vs, r) """
"""     if not same_direction(vt, vout, precision=1E-7): """
"""         print(f"Not same direction! vt={vt}, vout={vout}") """
"""         break """


# Testing self written cross product is correct
""" N = 10000 """
""" vecs = np.random.uniform(size=(N, 3)) """
""" for i in range(N-1): """
"""     v1 = np.cross(vecs[i], vecs[i+1]) """
"""     v2 = cross(vecs[i], vecs[i+1]) """
"""     if not np.allclose(v1, v2, atol=1E-10): """
"""         print(f"error: {v1}, {v2}") """
"""         break """

# Testing rotation around non-origin points
vecs = np.array(
    [
        [-1, 1, 1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [0, 0, 1],
    ]
)
point = np.array([-10, 0, 0])
r = get_rotation(X_, Y_)
print(vecs)
print("\n\n")
print(rotate_around(vecs, r, point))
