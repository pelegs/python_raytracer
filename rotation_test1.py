import numpy as np
import cv2
from libs.classes_cy import *
from libs.cymath import rotate_y


screen = Screen(resolution=(320, 240))


camera = Camera(screen=screen)
planet = Sphere(center=5*Y_, radius=1.5, color=BLUE, projected_color=BLUE)
moon = Sphere(center=3*Y_, radius=0.5, color=RED, projected_color=RED)

system = [planet, moon]
camera.draw_hittables(system, mask=False, samples=1)
cv2.imwrite(
    f"pics/rotation_test1.png",
    np.swapaxes(camera.screen.pixels, 0, 1),
)
