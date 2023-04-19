import pygame
from libs.classes import *


WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]


def draw_side(surface, side, color=WHITE, width=3):
    side_int = side.edges[:, :2].astype(np.int16)
    pygame.draw.line(surface, color, side_int[0], side_int[1], width=width)


vertices = np.zeros((3, 3))
vertices[:,:2] = np.random.randint(100, 500, size=(3, 2))
t = Triangle(vertices)

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Test of triangle methods")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # Get info
    x, y = pygame.mouse.get_pos()
    p = np.array([x, y, 0])
    if t.point_inside(p):
        tcol = GREEN
    else:
        tcol = WHITE

    # Draw stuff
    pygame.draw.circle(screen, RED, (x, y), 5)
    draw_side(screen, t.sides[0], tcol)
    draw_side(screen, t.sides[1], tcol)
    draw_side(screen, t.sides[2], tcol)
    
    pygame.display.flip()

pygame.quit()
