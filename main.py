import pygame
from src.game import SnakeGame
from src.controllers import KeyboardController

CELL = 20
TICK_RATE = 10

pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()

game = SnakeGame(20, 20)
controller = KeyboardController()

running = True
while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

    direction = controller.get_direction(events)
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
        pygame.draw.rect(
            screen,
            (0, 255, 0) if (x, y) == game.snake[0] else (0, 150, 0),
            (x * CELL, y * CELL, CELL, CELL),
        )

    pygame.display.flip()
    clock.tick(TICK_RATE)

pygame.quit()
