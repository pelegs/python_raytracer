import numpy as np
from libs.classes_cy import *

screen = Screen()
""" sc = np.random.uniform((0, 0), (1, screen.AR_inv), size=(20, 2)) """
wc = np.array([
    [-0.5, 1, screen.AR_half_inv],
    [0, 1, 0],
    [0.5, 1, -screen.AR_half_inv],
])
""" wc = screen.sc_to_wc(sc) """
sc2 = screen.wc_to_sc(wc)
print(screen.sc_to_pixels(sc2))
