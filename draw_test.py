import cv2
from libs.classes import *


screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)
sphere = Sphere()

camera.zoom(0.75)

num_frames = 60
q = np.array([0, 1, 0, 1.0/num_frames])
for frame in range(num_frames):
    if frame > 0:
        camera.rotate(q)
    """ sphere.center = camera.pos + 3*camera.d_scr """
    camera.draw_sphere(sphere)
    img = cv2.imwrite(f"frames/frame{frame:03d}.jpg", np.swapaxes(camera.screen.pixels, 0, 1))
