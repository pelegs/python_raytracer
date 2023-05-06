import cv2
from libs.mathlib import *
from libs.classes_cy import *


screen = Screen(resolution=(200, 100))
camera = Camera(screen=screen)

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

""" t1 = np.radians(45) """
""" q1 = rotation_x(t1) """
""" t2 = np.radians(30) """
""" q2 = rotation_y(t2) """
""" for triangle in triangles: """
"""     triangle.rotate(q2, point=np.mean(cube_points, axis=0)) """
"""     triangle.rotate(q1, point=np.mean(cube_points, axis=0)) """

camera.translate(10*Z_)

num_frames = 60
t3 = np.radians(360/num_frames)
q3 = rotation_y(t3)
frames = range(num_frames)
for frame in tqdm(frames, ):
    """ camera.project_triangles(triangles) """
    """ camera.projected_blur(n=5) """
    camera.apply_mask()
    img = cv2.imwrite(
        f"frames/anti_aliasing_colors_projected_{frame:03d}.png",
        np.swapaxes(camera.screen.projected, 0, 1)
    )
    for triangle in triangles:
        triangle.rotate(q3, point=np.mean(cube_points, axis=0))
    camera.draw_triangles(triangles, samples=1)
    img = cv2.imwrite(
        f"frames/anti_aliasing_colors_{frame:03d}.png",
        np.swapaxes(camera.screen.pixels, 0, 1)
    )
