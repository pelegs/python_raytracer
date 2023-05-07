import cv2
from libs.mathlib import *
from libs.classes_cy import *


screen = Screen(resolution=(640, 480))
camera = Camera(screen=screen)

cube_points = np.array(
    [
        [-1.34191, 1.0, -10.446395],
        [1.34191, 1.0, -9.553605],
        [0.44639, 1.0, -11.341913],
        [1.34191, 1.0, -9.553605],
        [-0.44639, -1.0, -8.658087],
        [1.34191, -1.0, -9.553605],
        [-0.44639, 1.0, -8.658087],
        [-1.34191, -1.0, -10.446395],
        [-0.44639, -1.0, -8.658087],
        [0.44639, -1.0, -11.341913],
        [-0.44639, -1.0, -8.658087],
        [-1.34191, -1.0, -10.446395],
        [0.44639, 1.0, -11.341913],
        [1.34191, -1.0, -9.553605],
        [0.44639, -1.0, -11.341913],
        [-1.34191, 1.0, -10.446395],
        [0.44639, -1.0, -11.341913],
        [-1.34191, -1.0, -10.446395],
        [-1.34191, 1.0, -10.446395],
        [-0.44639, 1.0, -8.658087],
        [1.34191, 1.0, -9.553605],
        [1.34191, 1.0, -9.553605],
        [-0.44639, 1.0, -8.658087],
        [-0.44639, -1.0, -8.658087],
        [-0.44639, 1.0, -8.658087],
        [-1.34191, 1.0, -10.446395],
        [-1.34191, -1.0, -10.446395],
        [0.44639, -1.0, -11.341913],
        [1.34191, -1.0, -9.553605],
        [-0.44639, -1.0, -8.658087],
        [0.44639, 1.0, -11.341913],
        [1.34191, 1.0, -9.553605],
        [1.34191, -1.0, -9.553605],
        [-1.34191, 1.0, -10.446395],
        [0.44639, 1.0, -11.341913],
        [0.44639, -1.0, -11.341913],
    ]
)

triangles = [
    Triangle(cube_points[n:n+3], id=i+1)
    for i, n in enumerate(np.arange(0, cube_points.shape[0], 3))
]
triangles[0].set_color(RED)
triangles[1].set_color(GREEN)
triangles[2].set_color(BLUE)
triangles[3].set_color(CYAN)
triangles[4].set_color(MAGENTA)
triangles[5].set_color(YELLOW)
triangles[6].set_color(RED)
triangles[7].set_color(GREEN)
triangles[8].set_color(BLUE)
triangles[9].set_color(CYAN)
triangles[10].set_color(MAGENTA)
triangles[11].set_color(YELLOW)

cube_center = np.mean(cube_points, axis=0)
t = np.radians(20)
s, c = np.sin(t), np.cos(t)
q = np.array([s, 0, 0, c])

for triangle in triangles:
    triangle.rotate(q, point=cube_center)

camera.project_triangles(triangles)
camera.projected_blur(n=5)
camera.apply_mask()
img = cv2.imwrite(
    f"pics/anti_aliasing_colors_projected.png",
    np.swapaxes(camera.screen.projected, 0, 1)
)
camera.draw_triangles(triangles, samples=10, mask=True)
img = cv2.imwrite(
    f"pics/anti_aliasing_colors.png",
    np.swapaxes(camera.screen.pixels, 0, 1)
)
