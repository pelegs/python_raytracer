import numpy as np
from itertools import product
from libs.classes import *


np.set_printoptions(precision=3, suppress=True)

w, h = 11, 17
screen = Screen(resolution=(w, h))
camera = Camera(screen=screen)
q = get_rotation(-Z_, Y_)
camera.rotate(q)

indx_i = [x for x in range(screen.resolution[0])]
indx_j = [x for x in range(screen.resolution[1])]
res = list(product(indx_i, indx_j))
points_wc = []
for p in res:
    points_wc.append(screen.get_pixel_center_wc(p))
points_wc = np.array(points_wc).reshape((w*h, 3))
points_wc = np.vstack((points_wc, screen.points_wcs))

np.save("coords", points_wc)
