import argparse
from enum import Enum

import matplotlib.pyplot as plt
import pygame
from IPython import display

from .controllers import AgentController, KeyboardController
from .game import SnakeGame, SnakeGameUpdateResult

plt.ion()  # type: ignore


class GameMode(Enum):
    USER = "user"
    REPLAY = "replay"
    AGENT = "agent"


CELL = 20
TICK_RATE = 30


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
    if visualize:
        screen = pygame.display.set_mode((400, 400))
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = None

    game = SnakeGame(20, 20)
    agent = AgentController()

    plot_scores: list[float] = []
    plot_mean_scores: list[float] = []
    total_score = 0
    high_score = 0
    frame_iteration = 0

    running = True

    while running:
        # Handle pygame events if visualizing
        if visualize:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

        # Get current state and choose action
        state_old = agent.get_state(game)
        relative_action = agent.get_action(state_old)
        absolute_direction = agent.convert_action_to_direction(relative_action, game.direction)

        # Perform action and get new state
        frame_iteration += 1
        update_result = game.update(absolute_direction)

        # Reward system
        reward = 0
        if update_result == SnakeGameUpdateResult.ATE_FOOD:
            reward = 10
        elif update_result == SnakeGameUpdateResult.GAME_OVER:
            reward = -10
        # Penalty if snake takes too long without eating (avoiding infinite loops)
        elif frame_iteration > 100 * len(game.snake):
            game.game_over = True
            reward = -10

        game_over = game.game_over
        score = len(game.snake)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, relative_action, reward, state_new, game_over)
        agent.remember(state_old, relative_action, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            frame_iteration = 0

            if score > high_score:
                high_score = score

            print("Game", agent.n_games, "Score", score, "High Score:", high_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            display.clear_output(wait=True)
            display.display(plt.gcf())  # type: ignore
            plt.clf()
            plt.title("Training...")  # type: ignore
            plt.xlabel("Number of Games")  # type: ignore
            plt.ylabel("Score")  # type: ignore
            plt.plot(plot_scores)  # type: ignore
            plt.plot(plot_mean_scores)  # type: ignore
            plt.ylim(ymin=0)  # type: ignore
            plt.text(len(plot_scores) - 1, plot_scores[-1], str(plot_scores[-1]))  # type: ignore
            plt.text(len(plot_mean_scores) - 1, plot_mean_scores[-1], str(plot_mean_scores[-1]))  # type: ignore
            plt.show(block=False)  # type: ignore
            plt.pause(0.1)

        # Render if visualization is enabled
        if visualize and screen is not None and clock is not None:
            screen.fill((0, 0, 0))

            # food
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
