import numpy as np
from libs.classes_cy import *
from libs.cymath import *


np.set_printoptions(precision=3, suppress=True)

screen = Screen()
print(screen.basis_vecs)
exit()
screen.rotate(
    axis=Y_,
    angle=np.radians(90)
)
transform_to_screen(
    screen.basis_vecs,
    screen.plane.normal,
)
