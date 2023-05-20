import numpy as np
from libs.classes import *


screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)

# shapes
sphere1 = Sphere(center=5 * Y_, radius=1.0, color=RED)
sphere2 = Sphere(center=5 * Y_ - 51 * Z_, radius=50.0, color=GREEN)
scene = [sphere1, sphere2]

""" camera.project_hittables(scene) """
""" camera.projected_blur(5) """
""" camera.apply_mask() """
camera.draw_hittables(scene, samples=30, mask=False)
camera.save_frame("pics/spheres.png")
