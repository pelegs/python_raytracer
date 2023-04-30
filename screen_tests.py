import numpy as np
from itertools import product
from libs.classes import *


np.set_printoptions(precision=3, suppress=True)

screen = Screen(resolution=(10, 5))
camera = Camera(screen=screen)
res = list(product(screen.resolution[0]))
exit()

points_wc = camera.screen.sc_to_wc(camera.screen.corners_scs)

np.save("coords", points_wc)
