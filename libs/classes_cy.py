import numpy as np
from .mathlib import *
from queue import PriorityQueue as pq
from tqdm import tqdm
from cymath import *


# Colors
BLACK = np.array([0, 0, 0])
WHITE = np.array([255, 255, 255])
BLUE = np.array([255, 0, 0])
GREEN = np.array([0, 255, 0])
RED = np.array([0, 0, 255])
CYAN = np.array([255, 255, 0])
MAGENTA = np.array([255, 0, 255])
YELLOW = np.array([0, 255, 255])
COLORS = [
    BLACK, WHITE, BLUE, GREEN, RED, CYAN, MAGENTA, YELLOW,
]
print(COLORS[0])


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
        return self.start + t * self.direction

    def intersects_sphere(self, sphere):
        oc = self.start - sphere.center
        a = norm2(self.direction)
        b = 2 * np.dot(oc, self.direction)
        c = norm2(oc) - sphere.radius_sqr
        D = b**2 - 4*a*c
        if D < 0:
            return -1
        else:
            return -(b + np.sqrt(D))/(2*a)

    def intersects_triangle(self, triangle):
        t = triangle.line_intersect(self)
        if t is not None:
            if triangle.point_inside(self.at_point(t)):
                return t
        return None

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
        normal = unit(cy_cross(v1, v2))
        return cls(normal, points[0])

    def get_normal_form(self):
        a, b, c = self.normal
        px, py, pz = self.point
        d = -(a * px + b * py + c * pz)
        self.normal_form = np.array([a, b, c, d])

    def reflect(self, direction):
        """Returns the direction of a line reflected from the plane."""
        return direction - 2 * (np.dot(direction, self.normal)) * self.normal

    def get_line_intersection(self, line):
        """
        Returns the point where line intersects the plane. If such point
        doesn't exist it return False.
        """
        return line_plane_intersection(
            line.start, line.direction, self.normal_form
        )

    def point_on_plane(self, p):
        return np.isclose(
            np.dot(self.normal, p), -1 * self.normal_form[3], PRECISION
        )

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

    def __init__(self, vertices, color=GREEN, id=-1):
        self.vertices = vertices
        self.create_sides()
        self.create_plane()
        self.color = color
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

    def point_inside(self, p):
        """Checks whether a given point p is inside the triangle."""
        return point_in_triangle(p, self.vertices)

    def line_intersect(self, line):
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

    def reflect(self, direction):
        """
        Returns the direction of a reflected line, assuming it intersects
        the triangle.
        """
        return self.plane.reflect(direction)

    def rotate(self, q, point=None):
        if point is None:
            rotation_center = self.center
        else:
            rotation_center = point
        self.vertices = rotate_around(self.vertices, q, rotation_center)
        self.create_sides()
        self.create_plane()

    def __str__(self):
        return f"""
        Vertices: {self.sides[0]}, {self.sides[1]}, {self.sides[2]}.
        Basis: {self.v1}, {self.v2}
        Plane: normal: {self.plane.normal}, point: {self.plane.point}.
        """


class Sphere:
    """TBW"""

    def __init__(self, center=-3*Z_, radius=1.0, color=RED):
        self.center = center
        self.radius = radius
        self.radius_sqr = radius**2
        self.color = color

    def point_inside(self, point):
        return distance2(point, self.center) <= self.radius_sqr

    def point_on(self, point):
        d2 = distance2(point, self.center)
        return np.isclose(d2, 0, PRECISION)

    def surface_normal(self, point):
        """
        Return the normal vector to the sphere at a given point on its surface.
        """
        """ if not self.point_on(point): """
        """     raise ValueError(f"Point {point} not on the surface of the sphere") """
        return unit(point - self.center)


class Screen:
    """
    TBW
    NOTE: except for final 3D transformations, all calculations are done in
    screen coordinates: x∈[0,1], y∈[0,1/aspect_ratio].
    """

    def __init__(self, resolution=VGA_480p_4_3):
        # resolution related
        self.aspect_ratio = resolution[0] / resolution[1]
        self.AR_ = 1.0 / self.aspect_ratio
        self.pixels = np.zeros(np.append(resolution, 3), dtype=np.int16)
        self.pixel_side = 1.0 / self.pixels.shape[0]
        self.resolution = self.pixels.shape

        # Screen coordinate system (scs)
        self.corners_scs = np.array(
            [
                # Corners order: NW, NE, SE, SW
                [0, 0],
                [1, 0],
                [1, self.AR_],
                [0, self.AR_],
            ]
        )
        self.center_scs = np.array([0.5, 0.5 * self.AR_])

        # World coordinate system (wcs)
        self.points_wcs = np.array(
            [
                # Corners order: NW, NE, SE, SW
                [-0.5, +0.5 * self.AR_, -1],
                [+0.5, +0.5 * self.AR_, -1],
                [+0.5, -0.5 * self.AR_, -1],
                [-0.5, -0.5 * self.AR_, -1],
                [0, 0, -1],
            ]
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
        dpxl = CORNERS_FROM_CENTER*self.pixel_side
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
        return self.sc_to_wc(self.rand_pts_in_pixel_sc(indices, n))
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

    def calc_screen_basis_vecs(self):
        """
        Calculates the 3 basis vectors of the screen in world coordinates:
        1. The vector from NW corner to NE corner.
        2. The vector from NW corner to SW corner scaled to length 1.
        3. The normal to above vectors.
        Since all three vectors have unit length and are orthonormal to each
        other, the resulting basis set is orthonormal.
        """
        u = self.points_wcs[1] - self.points_wcs[0]
        v = unit(self.points_wcs[3] - self.points_wcs[0])
        w = cy_cross(u, v)
        self.basis_vecs = np.array([u, v, w])

    def rotate(self, q, point=None):
        if point is None:
            point = self.points_wcs[4]
        self.points_wcs = rotate_around(self.points_wcs, q, point)
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
        sc_3 = np.c_[sc, np.zeros(n)]
        return np.dot(sc_3, self.basis_vecs) + self.points_wcs[0]


class Camera:
    """
    TBW
    d_scr: vector from camera position to screen center.
    """

    def __init__(
        self,
        pos=np.zeros(3),
        d_scr=-Z_,
        rotation=0.0,
        screen=Screen(),
    ):
        self.pos = pos
        self.d_scr = d_scr
        self.rotation = rotation
        self.screen = screen

    def rotate(self, q):
        """
        Rotates camera using the quaternion q.
        """
        self.d_scr = rotate_around(self.d_scr, q, self.pos)
        self.screen.rotate(q, self.pos)

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

    def dir_to_pixel_center(self, indices):
        """
        Get unit vector pointing from camera origin to the center of the
        pixel with given indices.
        """
        return unit(self.screen.get_pixel_center_wc(indices)-self.pos)

    def draw_triangles(self, triangles, samples=10):
        """
        This is just a test! Will be deleted later.
        """
        w, h = self.screen.resolution[:2]
        self.screen.pixels = np.zeros((w, h, 3), dtype=np.int16)
        for i in tqdm(range(self.screen.resolution[0]), leave=False):
            for j in range(self.screen.resolution[1]):
                rays = [
                    Ray(self.pos, screen_point)
                    for screen_point in self.screen.rand_pts_in_pixel_wc(
                        indices=(i, j), n=samples,
                    )
                ]
                pixel_color = np.zeros((samples, 3), dtype=np.int16)
                for k, ray in enumerate(rays):
                    for triangle in triangles:
                        t = ray.intersects_triangle(triangle)
                        if t is not None and t > 0:
                            ray.add_hit(t, triangle)
                    if ray.has_hits():
                        closest_triangle = ray.get_closest_hit()[1]
                        f = np.dot(
                            ray.direction, closest_triangle.plane.normal
                        )
                        pixel_color[k] = (
                            closest_triangle.color * np.abs(f)
                        ).astype(np.int16)
                self.screen.pixels[i, j] = np.mean(
                    pixel_color, axis=0
                ).astype(np.int16)


class Ray(Line):
    """docstring for Ray."""
    def __init__(self, start, direction):
        super(Ray, self).__init__(start, direction)
        self.hits = pq()
        self.color = BLACK

    def add_hit(self, t, object):
        try:
            self.hits.put((t, object), block=False)
        except:
            print(t, object.id)

    def has_hits(self):
        return not self.hits.empty()

    def get_closest_hit(self):
        if self.has_hits():
            return self.hits.get(block=False)


if __name__ == "__main__":
    pass
