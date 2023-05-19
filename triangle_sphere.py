import numpy as np
from libs.classes import *


screen = Screen(resolution=(160, 120))
camera = Camera(screen=screen)

# shapes
sphere = Sphere(center=5 * Y_, radius=1.0, color=RED)
triangle = Triangle(
    vertices=np.array(
        [
            [0, 7, 5],
            [-3, 7, -3],
            [3, 7, -3],
        ],
        dtype=np.double,
    ),
    color=GREEN,
)
scene = [triangle, sphere]

camera.project_hittables(scene)
camera.projected_blur(5)
camera.apply_mask()
camera.draw_hittables(scene, samples=1, mask=True)
camera.save_frame("pics/triangle_sphere.png")
