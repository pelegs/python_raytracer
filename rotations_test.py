import numpy as np
from libs.mathlib import *
""" from scipy.spatial.transform import Rotation as Rot """


np.set_printoptions(precision=3, suppress=True)

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


for _ in range(10000):
    vs, vt = np.random.uniform(-1, 1, (2, 3))
    r = get_rotation(vs, vt)
    vout = rotate_vecs(vs, r)
    if not same_direction(vt, vout, precision=1E-7):
        print(f"Not same direction! vt={vt}, vout={vout}")
        break
