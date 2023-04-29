import numpy as np
from libs.classes import *


np.set_printoptions(precision=3, suppress=True)

screen = Screen()
p = np.array(
    [
        # Corners order: NW, NE, SE, SW
        [0, 0],
        [1, 0],
        [1, screen.AR_],
        [0, screen.AR_],
        [0.5, 0.5 * screen.AR_],
    ]
)
q = get_rotation(-Z_, Y_)
screen.rotate(q)
screen.translate(Y_ + Z_)
print(screen.sc_to_wc(p))
