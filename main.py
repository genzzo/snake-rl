import pygame
from src.game import SnakeGame
from src.controllers import KeyboardController

CELL = 20

pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()

game = SnakeGame(20, 20)
controller = KeyboardController()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    direction = controller.get_direction()
    game.update(direction)

    if game.game_over:
        print("GAME OVER")
        running = False

    # ---- render ----
    screen.fill((0, 0, 0))

    # food
    if game.food is not None:
        fx, fy = game.food
        pygame.draw.rect(screen, (255, 0, 0), (fx * CELL, fy * CELL, CELL, CELL))

    # snake
    for x, y in game.snake:
        pygame.draw.rect(screen, (0, 255, 0), (x * CELL, y * CELL, CELL, CELL))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
