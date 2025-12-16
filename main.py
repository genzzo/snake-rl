import pygame
from typing import Callable, List
from enum import Enum
from src.game import SnakeGame
from src.controllers import KeyboardController


class GameType(Enum):
    USER = 0
    REPLAY = 1
    AGENT = 2


CELL = 20
TICK_RATE = 10
CURRENT_GAME_TYPE = GameType.USER


def run_user_game(
    game: SnakeGame, events: List[pygame.event.EventType], end_game: Callable[[], None]
):
    controller = KeyboardController()

    direction = controller.get_direction(events)
    game.update(direction)

    if game.game_over:
        print("GAME OVER")
        end_game()


def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    game = SnakeGame(20, 20)
    running: bool = True

    def end_game():
        nonlocal running
        running = False

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        if CURRENT_GAME_TYPE is GameType.USER:
            run_user_game(game, events, end_game)
        else:
            raise Exception("Game type is currently unsupported")

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


if __name__ == "__main__":
    main()
