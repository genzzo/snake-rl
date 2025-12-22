import argparse
from enum import Enum

import matplotlib.pyplot as plt
import pygame
from IPython import display

from .controllers import AgentController, KeyboardController, ReplayController
from .game import SnakeGame, SnakeGameUpdateResult
from .replay import ReplayRecorder

plt.ion()  # type: ignore


class GameMode(Enum):
    USER = "user"
    REPLAY = "replay"
    AGENT = "agent"


CELL = 20
TICK_RATE = 10000


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

    # Use agent's plot data (may be loaded from checkpoint)
    plot_scores = agent.plot_scores
    plot_mean_scores = agent.plot_mean_scores
    total_score = sum(plot_scores)  # Recalculate from loaded scores
    high_score = 0
    frame_iteration = 0

    # Initialize replay recorder
    replay_recorder = ReplayRecorder()
    save_replay_threshold = 5  # Save replays for scores >= this value

    # Start recording the first game
    replay_recorder.start_recording(
        game_number=agent.n_games + 1,
        grid_size=(game.width, game.height),
    )

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

        # Record frame for replay
        replay_recorder.record_frame(
            direction=game.direction,
            snake_positions=list(game.snake),
            food_position=game.food,
            score=len(game.snake),
        )

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

            # Stop replay recording
            replay_recorder.stop_recording()

            if score > high_score:
                high_score = score
                # Save checkpoint when achieving new high score
                agent.save_checkpoint(high_score=high_score)

            # Save replay for good games
            if score >= save_replay_threshold:
                replay_recorder.metadata["final_score"] = score
                replay_recorder.metadata["game_number"] = agent.n_games
                replay_recorder.metadata["is_high_score"] = score == high_score
                replay_recorder.save()

            # Start recording for next game
            replay_recorder.start_recording(
                game_number=agent.n_games + 1,
                grid_size=(game.width, game.height),
            )

            print("Game", agent.n_games, "Score", score, "High Score:", high_score)

            # Save checkpoint periodically (every 10 games)
            if agent.n_games % 10 == 0:
                agent.save_checkpoint(high_score=high_score)

            plot_scores.append(score)
            agent.plot_scores = plot_scores  # Keep agent's list in sync
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            agent.plot_mean_scores = plot_mean_scores  # Keep agent's list in sync

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


def run_replay_game(replay_file: str) -> None:
    """Run a game from a saved replay file."""
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    # Load replay
    metadata, frames = ReplayRecorder.load(replay_file)
    print(f"Loading replay: {replay_file}")
    print(f"Metadata: {metadata}")

    controller = ReplayController.from_replay_file(replay_file)
    game = SnakeGame(20, 20)

    running = True
    frame_idx = 0

    while running and frame_idx < len(frames):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        direction = controller.get_direction(events)
        game.update(direction)
        frame_idx += 1

        if game.game_over:
            print(f"GAME OVER - Final Score: {len(game.snake)}")
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--game-mode",
        type=str,
        choices=[mode.value for mode in GameMode],
        default=GameMode.AGENT.value,
        help="Type of game to run: 'user' for user-controlled, "
        "'agent' for AI agent-controlled, 'replay' for replay playback.",
    )

    parser.add_argument(
        "--replay-file",
        type=str,
        help="Path to replay file (required for replay mode)",
    )

    parser.add_argument(
        "--list-replays",
        action="store_true",
        help="List all available replays",
    )

    args = parser.parse_args()

    # Handle list replays
    if args.list_replays:
        replays = ReplayRecorder.list_replays()
        if replays:
            print("Available replays:")
            for replay in replays:
                print(f"  - {replay}")
        else:
            print("No replays found in ./replays/")
        return

    game_mode = GameMode(args.game_mode)

    if game_mode is GameMode.USER:
        run_user_game()
    elif game_mode is GameMode.AGENT:
        run_agent_game(False)
    elif game_mode is GameMode.REPLAY:
        if not args.replay_file:
            print("Error: --replay-file is required for replay mode")
            parser.print_help()
            return
        run_replay_game(args.replay_file)
    else:
        raise Exception("Game type is currently unsupported")


if __name__ == "__main__":
    main()
