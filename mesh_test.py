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
) + np.array([0, 0, -7])

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

t1 = np.radians(30)
q1 = rotation_y(t1)
cube.rotate(q1)
t2 = np.radians(30)
q2 = rotation_x(t2)
cube.rotate(q2)

camera.project_triangles(cube.faces)
camera.projected_blur(n=10)
cv2.imwrite(
    "pics/cube_as_mesh_projected.png",
    np.swapaxes(camera.screen.projected, 0, 1),
)
camera.apply_mask()
camera.draw_triangles(cube.faces)

cv2.imwrite(
    "pics/cube_as_mesh.png",
    np.swapaxes(camera.screen.pixels, 0, 1),
)
