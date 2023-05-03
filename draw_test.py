import cv2
from libs.classes import *


screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)
camera.translate(2*Z_)

points1 = np.array([[-2, -2, -4], [2, -2, -4], [0, 2, -4]])
points2 = np.array([[-2, 0, -6], [2, 0, -6], [0, 2, -6]])
points3 = np.array([[0, -1, -8], [2, 2, -7], [1, 4, -9]])
triangle1 = Triangle(points1, color=RED, id=1)
triangle2 = Triangle(points2, color=GREEN, id=2)
triangle3 = Triangle(points3, color=BLUE, id=3)

num_frames = 60
c = np.cos(np.pi/num_frames)
s = np.sin(np.pi/num_frames)

q1 = np.array([s, 0, 0, c])
q2 = np.array([0, s, 0, c])
q3 = np.array([0, 0, s, c])

frames = range(num_frames)
for frame in tqdm(frames):
    triangle1.rotate(q1)
    triangle2.rotate(q2)
    triangle3.rotate(q3)
    camera.draw_triangles([triangle1, triangle2, triangle3])
    img = cv2.imwrite(f"frames/triangles_{frame:03d}.png", np.swapaxes(camera.screen.pixels, 0, 1))
