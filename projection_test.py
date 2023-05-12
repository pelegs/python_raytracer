import numpy as np
from libs.classes_cy import *


screen = Screen(resolution=(640, 480))
camera = Camera(screen=screen)
sphere = Sphere(center=17*Y_, radius=1.5)
triangle = Triangle(
    np.array([
        [-1, 8, -1],
        [1, 8, -1],
        [0, 8, 2],
    ], dtype=np.double)
)
camera.project_hittable([sphere, triangle])
camera.projected_blur(5)
camera.save_projection("pics/sphere_projection_test.png")
camera.apply_mask()
camera.draw_hittables([sphere, triangle], samples=1, mask=True)
camera.save_frame("pics/sphere_test.png")
