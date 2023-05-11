import numpy as np
from libs.classes_cy import *
from libs.cymath import *


np.set_printoptions(precision=3, suppress=True)

screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)

N = 12
R = 5.0
angles = np.linspace(0, 2*np.pi, N+1)[:N]
centers = R*np.array([np.sin(angles), np.cos(angles), np.zeros(N)]).T
spheres = [
    Sphere(
        center=c, radius=0.75,
        color=np.random.randint(200, 256, size=3)
    )
    for i, c in enumerate(centers)
]

num_frames = 60
th = 2*np.pi/num_frames
for frame in tqdm(range(num_frames)):
    camera.draw_hittables(spheres, samples=1, mask=False)
    camera.save_frame(f"frames/spheres_test_{frame:03d}.png")
    camera.rotate(Z_, th)
