import numpy as np

""" from .mathlib import * """
from queue import PriorityQueue as pq
from tqdm import tqdm
import cv2
from random import choice as random_choice
from libs.cymath import *


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
COLORS = [
    BLACK,
    WHITE,
    GREY,
    BLUE,
    GREEN,
    RED,
    CYAN,
    MAGENTA,
    YELLOW,
]
PROJECTED_COLORS = [
    WHITE,
    GREY,
    BLUE,
    GREEN,
    RED,
    CYAN,
    MAGENTA,
    YELLOW,
]


class Line:
    """
    Represents a line in 3D space. (TBW)
    """

    def __init__(self, start, direction):
        self.start = start
        self.direction = unit(direction).flatten()

    @classmethod
    def from_two_points(cls, points):
        """Initiate via two points, preferably given as a 2x3 np array."""
        direction = points[1] - points[0]
        return cls(points[0], direction)

    def at_point(self, t):
        """
        The (parametric) line equation for the line is given by
        L(t) = start + direction*t. This function returns a point on the
        line given the parameter t.
        """
        return line_at_point(self.start, self.direction, t)

    def intersects_sphere(self, sphere):
        oc = self.start - sphere.center
        a = norm2(self.direction)
        b = 2 * np.dot(oc, self.direction)
        c = norm2(oc) - sphere.radius_sqr
        D = b**2 - 4 * a * c
        if D < 0:
            return -1
        else:
            return -(b + np.sqrt(D)) / (2 * a)

    def __str__(self):
        return f"start: {self.start}, direction: {self.direction}"


class Plane:
    """
    Represents a plane in 3D space.
    The plane is constructed from a direction (given by the normal vector)
    and a point on the plane. The normal form is a 4-vector with components
    (a, b, c, d), such that they solve the equation ax+by+cz+d=0.
    """

    def __init__(self, normal, point):
        self.normal = normal
        self.point = point
        self.get_normal_form()

    @classmethod
    def from_normal_form(cls, NFvec):
        """
        Instead of creating a plane using a normal and a point on the plane,
        we can create it using a normal form: ax+by+cz+d=0, s.t. a^2+b^2+c^2=1.
        Then (a,b,c) is the plane normal, but we still need to find a point on
        the plane. Assuming that the normal is not the zero vector (which is
        isn't allowed anyway), we can use its first non-zero component to find
        a point on the plane: let's say that the component is y, and so we take
        b. We then substitute x=0, z=0 into the normal form equation, which
        then gives by+d=0, i.e. y=-d/b. Thus, the point (0,-d/b,0) solves the
        normal form equation, and is thus guaranteed to be in the plane.
        """
        normal = NFvec[:3]
        # Get index of first non-zero element in normal
        if not NFvec[:3].any():
            raise ValueError("Normal vector can't be (0,0,0)!")
        i = (normal != 0).argmax()
        point = np.zeros(3)
        point[i] = -NFvec[3] / normal[i]
        return cls(normal, point)

    @classmethod
    def from_three_points(cls, points):
        """
        Initialize plane from three given points.
        """
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = unit(cross(v1, v2))
        return cls(normal, points[0])

    def get_normal_form(self):
        a, b, c = self.normal
        px, py, pz = self.point
        d = -(a * px + b * py + c * pz)
        self.normal_form = np.array([a, b, c, d])

    def reflect(self, point, direction):
        """Returns the direction of a line reflected from the plane."""
        normal = self.normal
        dn = rand_rotated_normal(normal, np.pi/6)
        normal = unit(normal + dn)
        return direction - 2 * (np.dot(direction, normal)) * normal

    def get_line_intersection(self, line):
        """
        Returns the point where line intersects the plane. If such point
        doesn't exist it return False.
        """
        return line_plane_intersection(line.start, line.direction, self.normal_form)

    def point_on(self, p):
        return point_on_plane(self.normal_form, p, 1.5e-5)

    def __str__(self):
        normal_txt = ",".join(map(str, self.normal))
        return f"Normal: {normal_txt}"


class Side(Line):
    """
    Represents a side of a polygon in 3D space.
    It is defined by two edges, such that edges[0] is the start point
    and edges[1] is the end point. The direction is given by the unit vector
    from the start pointing towards the end.
    NOTE: it is best to input the edges as a 2x3 numpy array: first row is
    edges[0], second row is edges[1].
    """

    def __init__(self, edges):
        self.edges = edges
        self.direction = unit(self.edges[1] - self.edges[0])
        super().__init__(self.edges[0], self.direction)
        self.length = distance(edges[1], edges[0])

    def __str__(self):
        return (
            f"Start: {self.edges[0]}, end: {self.edges[1]}, direction:"
            f"{self.direction}"
        )


class Triangle:
    """
    Represents a triangle in 3D space.
    It takes three 3D points as vertices, preferably as a 3x3 numpy array
    (s.t. each row represents a vertex). From this it creates the appropriate
    three sides of the triangle and the plane it lies on.
    """

    def __init__(self, vertices, color=GREEN, projected_color=GREEN, id=-1):
        self.vertices = vertices
        self.create_sides()
        self.create_plane()
        self.set_colors(color, projected_color)
        self.id = id

    def create_sides(self):
        # TODO: add case where sides are colinear
        pairs = np.array(
            [
                (a, b)
                for idx, a in enumerate(self.vertices)
                for b in self.vertices[idx + 1 :]
            ]
        )
        self.sides = [Side(pair) for pair in pairs]
        self.center = np.mean(self.vertices, axis=0)

    def create_plane(self):
        self.plane = Plane.from_three_points(self.vertices)

    def set_colors(self, color, projected_color):
        self.color = np.array(color, dtype=np.int16)
        self.projected_color = projected_color

    def point_inside(self, p):
        """Checks whether a given point p is inside the triangle."""
        return point_in_triangle(self.vertices, p)

    def get_line_intersection(self, line):
        """
        Checkes if line intersects the triangle's plane such that the
        intersection point is inside the triangle. If it does, the function
        returns the point of intersection. NOTE: right now returns the
        t argument of line at the intersection point.
        """
        t = self.plane.get_line_intersection(line)
        if t is not None:
            if self.point_inside(line.at_point(t)):
                return t
        return None

    def reflect(self, point, direction):
        """
        Returns the direction of a reflected line, assuming it intersects
        the triangle.
        """
        return self.plane.reflect(point, direction)

    def rotate(self, axis, point=None, angle=0.0):
        if point is None:
            rotation_center = self.center
        else:
            rotation_center = point
        self.vertices = rotate_around_point(self.vertices, axis, rotation_center, angle)
        self.create_sides()
        self.create_plane()

    def get_normal_at_point(self, p, theta=0.0):
        return rand_pt_solid_angle_rotated(self.plane.normal, theta)

    def draw_projection(self, camera, img):
        p_wc = triangle_projection(
            camera.pos, camera.screen.plane.normal_form,
            self.vertices
        )
        p_sc = camera.screen.wc_to_sc(p_wc)
        vertices_pixels = camera.screen.sc_to_pixels(p_sc).astype(np.int32)
        vertices_pixels[:, [0, 1]] = vertices_pixels[:, [1, 0]]
        return cv2.fillConvexPoly(
            img=camera.screen.projected,
            points=vertices_pixels,
            color=self.projected_color,
        )

    def ray_hits(self, ray):
        t = line_plane_intersection(ray.start, ray.direction, self.plane.normal_form)
        if t > 0:
            p = ray.at_point(t)
            if point_in_triangle(self.vertices, p):
                return t
        return False

    def __str__(self):
        return f"""
        Vertices: {self.sides[0]}, {self.sides[1]}, {self.sides[2]}.
        Basis: {self.v1}, {self.v2}
        Plane: normal: {self.plane.normal}, point: {self.plane.point}.
        """


class Sphere:
    """TBW"""

    def __init__(
        self,
        center=-3 * Z_,
        radius=1.0,
        color=RED,
        projected_color=RED,
    ):
        self.center = center
        self.radius = radius
        self.radius_sqr = radius**2
        self.color = np.array(color, dtype=np.uint8)
        if projected_color == "random":
            self.projected_color = random_choice(PROJECTED_COLORS)
        else:
            self.projected_color = projected_color

    def point_inside(self, point):
        return points_in_sphere(self.center, self.radius_sqr, point)

    def point_on(self, point):
        d2 = distance2(point, self.center)
        return np.isclose(d2, 0, PRECISION)

    def surface_normal(self, point):
        """
        Return the normal vector to the sphere at a given point on its surface.
        """
        return unit(point - self.center)

    def ray_hits(self, ray):
        return self.get_line_intersection(ray)

    def get_line_intersection(self, line):
        return line_sphere_intersection(
            line.start, line.direction, self.center, self.radius_sqr
        )

    def draw_projection(self, camera, img, color=WHITE):
        p, r_vis = sphere_projection(
            camera.pos, camera.screen.plane.normal_form, self.center,
            self.radius, camera.screen.width
        )
        p_sc = camera.screen.wc_to_sc(p.reshape(1, 3))
        p_pixel = camera.screen.sc_to_pixels(p_sc)[0][::-1]
        return cv2.circle(
            img=camera.screen.projected,
            center=p_pixel,
            radius=r_vis,
            color=self.projected_color,
            thickness=-1,
        )

    def get_normal_at_point(self, point, angle=0.05):
        norm = unit(point-self.center)
        return rand_pt_solid_angle_rotated(norm, angle)

    def reflect(self, point, direction, angle=0.05):
        normal = self.get_normal_at_point(point, angle=angle)
        return direction - 2 * (np.dot(direction, normal)) * normal


class Screen:
    """
    TBW
    NOTE: except for final 3D transformations, all calculations are done in
    screen coordinates: x∈[0,1], y∈[0,1/aspect_ratio].
    """

    def __init__(self, resolution=VGA_480p_4_3):
        # resolution related
        self.width = resolution[0] - 0.5
        self.aspect_ratio = resolution[0] / resolution[1]
        self.AR_inv = 1.0 / self.aspect_ratio
        self.AR_half_inv = 0.5 * self.AR_inv
        self.pixels = np.zeros(np.append(resolution, 3), dtype=np.uint8)
        self.pixel_side = 1.0 / self.pixels.shape[0]
        self.resolution = self.pixels.shape
        self.projected = np.ones(shape=self.pixels.shape, dtype=np.uint8) * 255
        self.non_zero_pixels = []

        # Screen coordinate system (scs)
        self.corners_scs = np.array(
            [
                # Corners order: NW, NE, SE, SW
                [0, 0],
                [1, 0],
                [1, self.AR_inv],
                [0, self.AR_inv],
            ],
            dtype=np.double,
        )
        self.center_scs = np.array([0.5, 0.5 * self.AR_inv], dtype=np.double)
        self.centering_sc = np.array([-0.5, self.AR_half_inv, 0])

        # World coordinate system (wcs)
        self.points_wcs = np.array(
            [
                # Corners order: NW, NE, SE, SW
                [-0.5, 1, self.AR_half_inv],
                [0.5, 1, self.AR_half_inv],
                [0.5, 1, -self.AR_half_inv],
                [-0.5, 1, -self.AR_half_inv],
                [0, 1, 0],
            ],
            dtype=np.double,
        )
        self.plane = Plane.from_three_points(self.points_wcs[:3])
        self.calc_screen_basis_vecs()

    def indices_check(self, indices):
        i, j = indices
        if not (0 <= i < self.resolution[0]):
            return f"""Index {i} greater than horizontal pixel range.
Allowed range is [0, {self.resolution[0]-1}]."""
        if not (0 <= j < self.resolution[1]):
            return f"""Index {j} greater than vertical pixel range.
Allowed range is [0, {self.resolution[1]-1}]."""
        return False

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv #
    # TODO: rewrite to accept an array of pixel indices
    def get_pixel_center_sc(self, indices):
        index_error = self.indices_check(indices)
        if index_error:
            raise ValueError(index_error)
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices]).flatten()
        return (indices + HALVES_2D) * self.pixel_side

    def get_pixel_corners_sc(self, indices):
        index_error = self.indices_check(indices)
        if index_error:
            raise ValueError(index_error)
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices]).flatten()
        dpxl = CORNERS_FROM_CENTER * self.pixel_side
        return self.get_pixel_center_sc(indices) + dpxl

    def get_pixel_center_wc(self, indices):
        point_sc = self.get_pixel_center_sc(indices)
        point_sc = np.array([point_sc])
        return self.sc_to_wc(point_sc)

    def get_pixel_corners_wc(self, indices):
        points_sc = self.get_pixel_corners_sc(indices)
        return self.sc_to_wc(points_sc)

    def rand_pts_in_pixel_sc(self, indices, n):
        corners_sc = self.get_pixel_corners_sc(indices)
        nw_corner_sc = corners_sc[0]
        se_corner_sc = corners_sc[2]
        return np.random.uniform(
            low=nw_corner_sc,
            high=se_corner_sc,
            size=(n, 2),
        )

    def rand_pts_in_pixel_wc(self, indices, n=10):
        rand_points = self.rand_pts_in_pixel_sc(indices, n)
        return self.sc_to_wc(rand_points)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

    def calc_screen_basis_vecs(self):
        """
        Calculates the 3 basis vectors of the screen in world coordinates:
        1. The vector from NW corner to NE corner.
        2. The vector from SW corner to NW corner scaled to length 1.
        3. The normal to above vectors.
        Since all three vectors have unit length and are orthonormal to each
        other, the resulting basis set is orthonormal.
        Note: the set is technically right-handed.
        """
        e1 = unit(self.points_wcs[1] - self.points_wcs[0])
        e2 = unit(self.points_wcs[0] - self.points_wcs[3])
        n = cross(e1, e2)
        self.basis_vecs = np.array([e1, e2, n])
        self.basis_vecs_inv = np.linalg.inv(self.basis_vecs)

    def rotate(self, axis, point=None, angle=0.0):
        if point is None:
            point = self.points_wcs[4]
        self.points_wcs = rotate_around_point(self.points_wcs, axis, point, angle)
        self.plane = Plane.from_three_points(self.points_wcs[:3])
        self.calc_screen_basis_vecs()

    def translate(self, dr):
        self.points_wcs = self.points_wcs + dr
        self.plane = Plane.from_three_points(self.points_wcs[:3])
        self.calc_screen_basis_vecs()

    def sc_to_wc(self, sc):
        """
        Input: point(s) in screen coordinates.
        Output: point(s) in world coordinates.
        NOTE: maybe add limitation on values s.t. sc is indeed inside screen?
        """
        n = sc.shape[0]
        sc_3 = np.c_[sc, np.zeros(n)] * FLIP_Y + self.centering_sc
        return np.dot(sc_3, self.basis_vecs) + self.points_wcs[4]

    def wc_to_sc(self, wc):
        sc = wc - self.points_wcs[4]
        sc = np.dot(sc, self.basis_vecs_inv)
        sc -= self.centering_sc
        sc = sc * FLIP_Y
        return sc[:, :2]

    def sc_to_pixels(self, sc):
        return (sc * self.width).astype(np.int16)


class Camera:
    """
    TBW
    d_scr: vector from camera position to screen center.
    """

    def __init__(
        self,
        pos=np.zeros(3),
        d_scr=-Z_.reshape((1, 3)),
        rotation=0.0,
        screen=Screen(),
    ):
        self.pos = pos
        self.d_scr = d_scr
        self.rotation = rotation
        self.screen = screen

    def rotate(self, axis, angle):
        """
        Rotates camera using the quaternion q.
        """
        self.d_scr = rotate_around_point(self.d_scr, axis, self.pos, angle)
        self.screen.rotate(axis, self.pos, angle)

    def translate(self, dr):
        """
        Translate the camera by dr.
        """
        self.pos = self.pos + dr
        self.screen.translate(dr)

    def zoom(self, factor):
        if factor == 0:
            raise ValueError("Can't scale distance to screen by zero!")
        d_scr_old = self.d_scr
        self.d_scr = self.d_scr * factor
        self.screen.translate(self.d_scr - d_scr_old)

    def look_at(self, point):
        direction = point - self.pos
        axis, angle = get_axis_angle(self.screen.basis_vecs[2], direction)
        self.rotate(axis, angle)

    def dir_to_pixel_center(self, indices):
        """
        Get unit vector pointing from camera origin to the center of the
        pixel with given indices.
        """
        return unit(self.screen.get_pixel_center_wc(indices) - self.pos)

    def project_hittables(self, hittables):
        self.screen.projected = np.zeros(shape=self.screen.pixels.shape, dtype=np.uint8)
        for hittable in hittables:
            self.screen.projected = hittable.draw_projection(
                self, self.screen.projected
            )

    def projected_blur(self, n=3):
        kernel = np.ones((n, n), dtype=np.float32) / n**2
        self.screen.projected = cv2.filter2D(self.screen.projected, -1, kernel)

    def apply_mask(self):
        gray = cv2.cvtColor(self.screen.projected, cv2.COLOR_BGR2GRAY)
        self.screen.non_zero_pixels = np.column_stack(np.where(gray > 3))

    def draw_hittables(self, hittable_list, samples=10, mask=True):
        """
        This is just a test! Will be deleted later.
        """
        w, h = self.screen.resolution[:2]
        self.screen.pixels = np.zeros((w, h, 3), dtype=np.uint8)
        if mask:
            process = self.screen.non_zero_pixels
        else:
            process = [(i, j) for i in range(w) for j in range(h)]
        for i, j in tqdm(process, leave=False):
            rays = [
                Ray(self.pos, screen_point, max_bounces=20)
                for screen_point in self.screen.rand_pts_in_pixel_wc(
                    indices=(i, j),
                    n=samples,
                )
            ]
            pixel_color = np.zeros((samples, 3))
            for k, ray in enumerate(rays):
                ray.calc_hits(hittable_list)
                pixel_color = np.vstack((pixel_color, ray.color))
            self.screen.pixels[i, j] = np.mean(pixel_color, axis=0).astype(np.int)

    def save_frame(self, filename):
        cv2.imwrite(
            filename,
            np.swapaxes(self.screen.pixels, 0, 1),
        )

    def save_projection(self, filename):
        cv2.imwrite(
            filename,
            np.swapaxes(self.screen.projected, 0, 1),
        )


class Ray(Line):
    """docstring for Ray."""

    def __init__(self, start, direction, max_bounces=3):
        super(Ray, self).__init__(start, direction)
        self.hits = pq()
        self.max_bounces = max_bounces
        self.num_bounces = 0
        self.active = True

        # Make nice bg
        s = np.array([255, 195, 120])
        t = np.array([255, 240, 240])
        y = self.direction[1]
        self.color = ((1-y)*t + y*s).astype(np.int16)

    def add_hit(self, t, object):
        try:
            self.hits.put((t, object), block=False)
            self.color = np.vstack((self.color, object.color))
        except:
            print("error:", t, object.id)

    def has_hits(self):
        return not self.hits.empty()

    def reset_hits(self):
        with self.hits.mutex:
            self.hits.queue.clear()

    def reset_color(self):
        self.color = np.zeros(3)

    def calc_avg_color(self):
        n = self.color.shape[0]
        ws = normed_geo_weights(n)
        self.color = np.average(self.color, axis=0, weights=ws)
        return self.color

    def get_closest_hit(self):
        if self.has_hits():
            return self.hits.get(block=False)
        else:
            return -1, None

    def bounce(self, pos, normal):
        self.start = pos
        if norm(normal) != 1.0:
            normal = unit(normal)
        self.direction = reflect(self.direction, normal)
        self.num_bounces += 1

    def calc_hits(self, hittables):
        closest_hittable = None
        while self.num_bounces < self.max_bounces and self.active:
            self.reset_hits()
            for hittable in hittables:
                t = hittable.ray_hits(self)
                if t is not None and t>0:
                    self.add_hit(t, hittable)
            t, closest_hittable = self.get_closest_hit()
            if closest_hittable is not None:
                hit_point = self.at_point(t)
                normal = hittable.get_normal_at_point(hit_point, np.pi/2)
                self.bounce(hit_point, normal)
            else:
                self.active = False
        if self.num_bounces > 0:
            f = 0.5**(self.num_bounces-1)
            normed_color = np.sqrt(f*np.array([0.5, 0.5, 0.5]))
            self.color = (normed_color*255).astype(np.int)


class Mesh:
    """
    TBW
    """

    def __init__(self, faces, color=GREY, id=-1):
        self.faces = faces
        self.color = color
        self.id = id
        self.center = np.mean([face.center for face in self.faces], axis=0)

    @classmethod
    def from_vertices(cls, vertices, color=GREY, id=-1):
        triangles = [
            Triangle(vertices[n : n + 3], id=i)
            for i, n in enumerate(np.arange(0, vertices.shape[0], 3))
        ]
        return cls(triangles, color, id)

    def rotate(self, axis, point=None, angle=0.0):
        if point is None:
            point = self.center
        for face in self.faces:
            face.rotate(axis, point, angle)

    def translate(self, vec):
        for face in self.faces:
            face.translate(vec)

    def color_randomly(self, colors_list=[RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW]):
        for face1, face2 in zip(self.faces[::2], self.faces[1::2]):
            random_color = random_choice(colors_list)
            face1.set_color(random_color)
            face2.set_color(random_color)


# Not a class, I know. Will correct later
def factor_color(color, factor):
    return (np.array(color).astype(np.float)*factor).astype(np.int16)


if __name__ == "__main__":
    pass
