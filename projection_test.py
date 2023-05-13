import numpy as np
from libs.classes_cy import *


screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)
R = 1.5E0
sphere = Sphere(center=10*Y_, radius=1.5, color=RED)
ground = Sphere(center=-R*Z_+10*Y_, radius=R, color=GREY)
""" camera.project_hittable([sphere, ground]) """
""" camera.projected_blur(5) """
""" camera.save_projection("pics/sphere_projection_test.png") """
""" camera.apply_mask() """
camera.draw_hittables([sphere, ground], samples=1, mask=False)
camera.save_frame("pics/sphere_test.png")
