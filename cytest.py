import numpy as np
from libs.classes_cy import *


points = np.array(
    [
        [1, -2, 10],
        [2, 0, 10],
        [0, 2, 10],
    ],
    dtype=np.double
)
triangle = Triangle(points)
screen = Screen(resolution=(10, 10))
camera = Camera(screen=screen)

camera.project_triangles([triangle])
print(camera.screen.projected)
