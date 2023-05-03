import cv2
from libs.classes import *


screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)
camera.translate(5*Z_)
points = np.array([
    [-2, -2, -4],
    [2, -2, -4],
    [0, 2, -4],
])
triangle = Triangle(points)

num_frames = 180
frames = np.arange(0, num_frames)
q = np.array([0, 1, 0, np.pi/num_frames])
for frame in tqdm(frames):
    triangle.rotate(q)
    camera.draw_triangle(triangle)
    img = cv2.imwrite(f"frames/triangle_{frame:03d}.png", np.swapaxes(camera.screen.pixels, 0, 1))
