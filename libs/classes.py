import numpy as np
from itertools import product
from .mathlib import *


class Line:
    """
    Represents a line in 3D space. (TBW)
    """
    def __init__(self, start, direction):
        self.start = start
        self.direction = unit(direction)

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
        normal = unit(np.cross(v1, v2))
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
        if np.dot(self.normal, line.direction) == 0:
            return None
        else:
            a, b, c, d = self.normal_form
            sx, sy, sz = line.start
            vx, vy, vz = line.direction
            t = -(a * sx + b * sy + c * sz + d) / (a * vx + b * vy + c * vz)
            return line.at_point(t)

    def point_on_plane(self, p):
        return np.isclose(np.dot(self.normal, p), -1 * self.normal_form[3], PRECISION)

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

    def __init__(self, vertices):
        self.vertices = vertices
        self.create_sides()
        self.create_plane()

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

    def create_plane(self):
        self.plane = Plane.from_three_points(self.vertices)

    def point_inside(self, p):
        """Checks whether a given point p is inside the triangle."""
        trans_triangle = self.vertices - p
        u = np.cross(trans_triangle[1], trans_triangle[2])
        v = np.cross(trans_triangle[2], trans_triangle[0])
        w = np.cross(trans_triangle[0], trans_triangle[1])
        if np.dot(u, v) < 0:
            return False
        if np.dot(u, w) < 0:
            return False
        return True

    def line_intersect(self, line):
        """
        Checkes if line intersects the triangle's plane such that the
        intersection point is inside the triangle. If it does, the function
        returns the point of intersection.
        """
        p = self.plane.get_line_intersection(line)
        if p and self.point_inside(p):
            return p
        return False

    def reflect(self, direction):
        """
        Returns the direction of a reflected line, assuming it intersects
        the triangle.
        """
        return self.plane.reflect(direction)

    def __str__(self):
        return f"""
        Vertices: {self.sides[0]}, {self.sides[1]}, {self.sides[2]}.
        Basis: {self.v1}, {self.v2}
        Plane: normal: {self.plane.normal}, point: {self.plane.point}.
        """


class Sphere:
    """TBW"""

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.radius_sqr = radius**2

    def point_inside(self, point):
        return distance2(point, self.center) <= self.radius_sqr

    def point_on(self, point):
        d2 = distance2(point, self.center)
        return np.isclose(d2, 0, PRECISION)

    def surface_normal(self, point):
        """
        Return the normal vector to the sphere at a given point on its surface.
        """
        if not self.point_on(point):
            raise ValueError(f"Point {point} not on the surface of the sphere")
        return unit(point - self.center)


class Screen:
    """
    TBW
    NOTE: except for final 3D transformations, all calculations are done in
    screen coordinates: x∈[0,1], y∈[0,1/aspect_ratio].
    """
    def __init__(self, resolution=VGA_480p_4_3):
        self.aspect_ratio = resolution[0] / resolution[1]
        self.AR_ = 1.0 / self.aspect_ratio
        self.pixels = np.zeros(np.append(resolution, 3))
        self.pixel_side = 1.0 / self.pixels.shape[0]
        self.resolution = self.pixels.shape
        self.corners = np.array(
            [
                # Corners order: NW, NE, SE, SW
                [0, 0],
                [1, 0],
                [1, self.AR_],
                [0, self.AR_],
            ]
        )
        """ self.plane = Plane.from_ """
        self.center = np.array([0.5, 0.5 * self.AR_])

    def get_pixel_center(self, indices):
        i, j = indices
        if not (0 <= i < self.resolution[0]):
            raise ValueError(
                f"""Index {i} greater than horizontal pixel range.
                Allowed range is [0, {self.resolution[0]-1}].
                """
            )
        if not (0 <= j < self.resolution[1]):
            raise ValueError(
                f"""Index {j} greater than vertical pixel range.
                Allowed range is [0, {self.resolution[1]-1}].
                """
            )
        if not isinstance(indices, np.ndarray):
            indices = np.array([indices]).flatten()
        return (indices + np.array([0.5, 0.5])) * self.pixel_side


class Camera:
    """
    TBW
    """
    def __init__(
        self,
        pos=np.zeros(3),
        dir=-Z_DIR,
        rotation=0.0,
        view_angle=30.0,
        aspect_ratio=1.0,
    ):
        self.pos = pos
        self.dir = dir
        self.rotation = rotation
        self.view_angle = view_angle
        self.aspect_ratio = aspect_ratio


if __name__ == "__main__":
    pass
