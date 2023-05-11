import numpy as np
from libs.classes_cy import *


screen = Screen(resolution=(160, 120))
camera = Camera(screen=screen)

# shapes
sphere = Sphere(center=5 * Y_, radius=1.0, color=RED)
triangle = Triangle(
    vertices=np.array(
        [
            [0, 100, -1],
            [-100, -10, -1],
            [100, -10, -1],
        ],
        dtype=np.double,
    ),
    color=GREEN,
)

camera.draw_hittables([sphere, triangle], samples=10, mask=False)
camera.save_frame("pics/triangle_sphere.png")
