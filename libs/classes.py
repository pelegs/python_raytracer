import numpy as np
from .mathlib import *


class Side:
    """To be written"""

    def __init__(self, edges):
        self.edges = edges
        self.direction = unit(self.edges[1] - self.edges[0])

    def point_on(self, t):
        return self.edges[0] + t * self.direction

    def __str__(self):
        return f"Start: {self.edges[0]}, end: {self.edges[1]}, direction: {self.direction}"


class Triangle:
    """To be written"""

    def __init__(self, vertices):
        self.vertices = vertices
        self.create_sides()

    def create_sides(self):
        pairs = np.array([
            (a, b)
            for idx, a in enumerate(self.vertices)
            for b in self.vertices[idx+1:]
        ])
        self.sides = [Side(pair) for pair in pairs]

    def point_inside(self, p):
        trans_triangle = self.vertices - p
        u = np.cross(trans_triangle[1], trans_triangle[2])
        v = np.cross(trans_triangle[2], trans_triangle[0])
        w = np.cross(trans_triangle[0], trans_triangle[1])
        if np.dot(u, v) < 0:
            return False
        if np.dot(u, w) < 0:
            return False
        return True


if __name__ == "__main__":
    pass
