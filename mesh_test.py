import cv2
from libs.mathlib import *
from libs.classes_cy import *


screen = Screen(resolution=(320, 240))
camera = Camera(screen=screen)

vertices = np.array(
    [
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
    ]
) + np.array([0, 0, -4])

cube_triangle_vertices = np.array([
    vertices[0], vertices[1], vertices[2],
    vertices[0], vertices[3], vertices[2],
    vertices[4], vertices[5], vertices[6],
    vertices[4], vertices[7], vertices[6],
    vertices[1], vertices[5], vertices[6],
    vertices[1], vertices[2], vertices[6],
    vertices[0], vertices[4], vertices[7],
    vertices[0], vertices[3], vertices[7],
    vertices[0], vertices[4], vertices[5],
    vertices[0], vertices[1], vertices[5],
    vertices[3], vertices[7], vertices[6],
    vertices[3], vertices[2], vertices[6],
])

cube = Mesh.from_vertices(cube_triangle_vertices)
cube.color_randomly()
cube.rotate(
    axis=X_, point=None, angle=np.pi/6
)

""" camera.rotate(Y_, np.pi/24) """
# TODO: projection after camera rotation needs work!

num_frames = 60
for frame in tqdm(range(num_frames)):
    cube.rotate(
        axis=Y_, point=None, angle=2*np.pi/num_frames
    )

    camera.project_triangles(cube.faces)
    camera.projected_blur(n=10)
    cv2.imwrite(
        f"frames/cube_as_mesh_projected_{frame:03d}.png",
        np.swapaxes(camera.screen.projected, 0, 1),
    )
    camera.apply_mask()
    camera.draw_triangles(cube.faces, samples=5)
    cv2.imwrite(
        f"frames/cube_as_mesh_{frame:03d}.png",
        np.swapaxes(camera.screen.pixels, 0, 1),
    )
