import numpy as np
import cv2
from libs.classes_cy import *


screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)
planet = Sphere(center=-4*Z_, radius=1.0, color=BLUE, projected_color=BLUE)
moon = Sphere(center=-5.5*Z_, radius=.5, color=RED, projected_color=RED)
dr = 1.5

system = [planet, moon]

num_frames = 30
frames = np.arange(num_frames)
ts = np.linspace(0, 2*np.pi, num_frames+1)
for t, frame in tqdm(zip(ts, frames)):
    moon.center = planet.center + dr*np.array([np.cos(t), 0.0, np.sin(t)])
    camera.project_hittable(system)
    cv2.imwrite(
        f"frames/planet_moon_projection_{frame:03d}.png",
        np.swapaxes(camera.screen.projected, 0, 1),
    )
    """ camera.draw_hittables([planet, moon], mask=False) """
    """ cv2.imwrite( """
    """     f"frames/planet_moon_{frame:03d}.png", """
    """     np.swapaxes(camera.screen.pixels, 0, 1), """
    """ ) """
