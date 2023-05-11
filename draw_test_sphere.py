import numpy as np
import cv2
from libs.classes_cy import *
from libs.cymath import rotate_y


screen = Screen(resolution=(320, 240))


camera = Camera(screen=screen)
planet = Sphere(center=5*Y_, radius=1.5, color=BLUE, projected_color=BLUE)
moon = Sphere(center=Y_, radius=0.5, color=RED, projected_color=RED)
dr = 2

system = [planet, moon]

num_frames = 30
frames = np.arange(num_frames)
ts = np.linspace(0, 2*np.pi, num_frames+1)
for t, frame in tqdm(zip(ts, frames)):
    moon.center = planet.center + rotate_y(dr*np.array([[np.cos(t), np.sin(t), 0.0]]), np.pi/5)[0]
    """ camera.project_hittable(system) """
    """ cv2.imwrite( """
    """     f"frames/planet_moon_projection_{frame:03d}.png", """
    """     np.swapaxes(camera.screen.projected, 0, 1), """
    """ ) """
    camera.draw_hittables(system, mask=False, samples=1)
    cv2.imwrite(
        f"frames/planet_moon_{frame:03d}.png",
        np.swapaxes(camera.screen.pixels, 0, 1),
    )
