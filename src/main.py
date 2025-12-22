import argparse
from enum import Enum

import pygame

from .controllers import KeyboardController
from .game import SnakeGame, SnakeGameUpdateResult


class GameMode(Enum):
    USER = "user"
    REPLAY = "replay"
    AGENT = "agent"


CELL = 20
TICK_RATE = 10


def run_user_game() -> None:
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


def run_agent_game(visualize: bool = False) -> None:
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    game = SnakeGame(20, 20)
    controller = KeyboardController()  # temp until we create agent
    running = True

    reward = 0
    frame_iteration = 0

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        frame_iteration += 1
        direction = controller.get_direction(events)
        update_result = game.update(direction)

        if update_result is SnakeGameUpdateResult.ATE_FOOD:
            reward += 10
        elif update_result is SnakeGameUpdateResult.GAME_OVER:
            reward -= 10
        # snake is not eating and just going around
        elif frame_iteration > 100 * len(game.snake):
            reward = -10

        if game.game_over:
            print(f"GAME OVER. User simulated agent result finished with reward: {reward}")
            running = False

        # ---- render ----
        if visualize:
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--game-mode",
        type=str,
        choices=[mode.value for mode in GameMode],
        default=GameMode.AGENT.value,
        help="Type of game to run: 'user' for user-controlled, 'agent' for AI agent-controlled.",
    )

    args = parser.parse_args()
    game_mode = GameMode(args.game_mode)

    if game_mode is GameMode.USER:
        run_user_game()
    elif game_mode is GameMode.AGENT:
        run_agent_game(True)
    else:
        raise Exception("Game type is currently unsupported")


if __name__ == "__main__":
    main()
