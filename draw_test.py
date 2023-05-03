import cv2
from libs.classes import *


screen = Screen(resolution=(160, 120))
camera = Camera(screen=screen)
camera.translate(2*Z_)
points = np.array([
    [-2, -2, -4],
    [2, -2, -4],
    [0, 2, -4],
    [-4, 1, -9],
    [-1, 3, -10],
    [-2, -3, -9],
    [-2, 0, -21],
    [2, -1, -20],
    [0, 4, -18],
])
triangle1 = Triangle(points[:3], color=RED, id=1)
triangle2 = Triangle(points[3:6], color=GREEN, id=2)
triangle3 = Triangle(points[6:], color=BLUE, id=3)

num_frames = 180

q1 = np.array([1, 0, 0, np.pi])
q2 = np.array([0, 1, 0, np.pi])
q3 = np.array([0, 0, 1, np.pi])

frames = range(num_frames)
for frame in tqdm(frames):
    """ triangle1.rotate(q1) """
    """ triangle2.rotate(q2) """
    """ triangle3.rotate(q3) """
    camera.translate(-0.1*Z_+0.025*X_)
    camera.draw_triangles([triangle1, triangle2, triangle3])
    img = cv2.imwrite(f"frames/triangles_{frame:03d}.png", np.swapaxes(camera.screen.pixels, 0, 1))
