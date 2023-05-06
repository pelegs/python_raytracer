import cv2
from libs.mathlib import *
from libs.classes_cy import *


""" screen = Screen(resolution=(640, 480)) """
screen = Screen(resolution=(200, 100))
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
    Triangle(cube_points[n:n+3], id=i+1)
    for i, n in enumerate(np.arange(0, cube_points.shape[0], 3))
]
triangles[0].color = RED
triangles[1].color = GREEN
triangles[2].color = BLUE
triangles[3].color = CYAN
triangles[4].color = MAGENTA
triangles[5].color = YELLOW
triangles[6].color = RED
triangles[7].color = GREEN
triangles[8].color = BLUE
triangles[9].color = CYAN
triangles[10].color = MAGENTA
triangles[11].color = YELLOW

t1 = np.radians(45)
q1 = rotation_x(t1)
t2 = np.radians(30)
q2 = rotation_y(t2)
for triangle in triangles:
    triangle.rotate(q2, point=np.mean(cube_points, axis=0))
    triangle.rotate(q1, point=np.mean(cube_points, axis=0))

num_frames = 60
t3 = np.radians(360/num_frames)
q3 = rotation_y(t3)
frames = range(num_frames)
for frame in tqdm(frames):
    for triangle in triangles:
        triangle.rotate(q3, point=np.mean(cube_points, axis=0))
    camera.draw_triangles(triangles, samples=8)
    img = cv2.imwrite(
        f"frames/anti_aliasing_colors{frame:03d}.png",
        np.swapaxes(camera.screen.pixels, 0, 1)
    )
