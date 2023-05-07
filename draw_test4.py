import cv2
from libs.mathlib import *
from libs.classes_cy import *


screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)

points = np.array([
    [-3, -2, -9],
    [1, -2, -9],
    [-1, 2, -9],
    [1, -2, -9],
    [5, -2, -9],
    [3, 2, -9],
    [-1, 0, -12],
    [3, 0, -12],
    [1, 4, -12],
], dtype=np.double)
triangle1 = Triangle(points[:3], color=RED)
triangle2 = Triangle(points[3:6], color=BLUE)
triangle3 = Triangle(points[6:], color=GREEN)
triangles = [triangle1, triangle2, triangle3]

camera.project_triangles(triangles)
camera.projected_blur(n=15)
camera.apply_mask()
img = cv2.imwrite(
    f"pics/debug01_proj.png",
    np.swapaxes(camera.screen.projected, 0, 1)
)

camera.draw_triangles(triangles, samples=10, mask=True)
img = cv2.imwrite(
    f"pics/debug01.png",
    np.swapaxes(camera.screen.pixels, 0, 1)
)
