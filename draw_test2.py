import cv2
from libs.mathlib import *
from libs.classes import *


screen = Screen(resolution=(640, 480))
camera = Camera(screen=screen)
camera.translate(0.25 * Z_)
""" t = np.radians(-10) """
""" q = rotation_x(t) """
""" camera.rotate(q) """

cube_points = np.array(
    [
        [-1.34191, 1.0, -4.446395],
        [1.34191, 1.0, -3.553605],
        [0.44639, 1.0, -5.341913],
        [1.34191, 1.0, -3.553605],
        [-0.44639, -1.0, -2.658087],
        [1.34191, -1.0, -3.553605],
        [-0.44639, 1.0, -2.658087],
        [-1.34191, -1.0, -4.446395],
        [-0.44639, -1.0, -2.658087],
        [0.44639, -1.0, -5.341913],
        [-0.44639, -1.0, -2.658087],
        [-1.34191, -1.0, -4.446395],
        [0.44639, 1.0, -5.341913],
        [1.34191, -1.0, -3.553605],
        [0.44639, -1.0, -5.341913],
        [-1.34191, 1.0, -4.446395],
        [0.44639, -1.0, -5.341913],
        [-1.34191, -1.0, -4.446395],
        [-1.34191, 1.0, -4.446395],
        [-0.44639, 1.0, -2.658087],
        [1.34191, 1.0, -3.553605],
        [1.34191, 1.0, -3.553605],
        [-0.44639, 1.0, -2.658087],
        [-0.44639, -1.0, -2.658087],
        [-0.44639, 1.0, -2.658087],
        [-1.34191, 1.0, -4.446395],
        [-1.34191, -1.0, -4.446395],
        [0.44639, -1.0, -5.341913],
        [1.34191, -1.0, -3.553605],
        [-0.44639, -1.0, -2.658087],
        [0.44639, 1.0, -5.341913],
        [1.34191, 1.0, -3.553605],
        [1.34191, -1.0, -3.553605],
        [-1.34191, 1.0, -4.446395],
        [0.44639, 1.0, -5.341913],
        [0.44639, -1.0, -5.341913],
    ]
)

triangles = [
    Triangle(cube_points[n : n + 3], color=RED, id=i + 1)
    for i, n in enumerate(np.arange(0, 36, 3))
]

t1 = np.radians(45)
q1 = rotation_x(t1)
t2 = np.radians(30)
q2 = rotation_y(t2)
for triangle in triangles:
    triangle.rotate(q2, point=np.mean(cube_points, axis=0))
    triangle.rotate(q1, point=np.mean(cube_points, axis=0))

camera.draw_triangles(triangles, samples=10)
img = cv2.imwrite(f"anti_aliasing_test.png", np.swapaxes(camera.screen.pixels, 0, 1))
