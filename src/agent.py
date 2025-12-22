from .controllers import AgentController
from .game import SnakeGame


def train_agent(game: SnakeGame) -> None:
    plot_scores: list[float] = []
    plot_mean_scores: list[float] = []
    total_score = 0
    high_score = 0

    agent = AgentController()

    while True:
        state_old = agent.get_state(game)

        relative_action = agent.get_action(state_old)
        absolute_direction = agent.convert_action_to_direction(relative_action, game.direction)

        game.update(absolute_direction)
        # TODO: fix reward system
        reward, game_over, score = (0, game.game_over, len(game.snake))

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, relative_action, reward, state_new, game_over)

        agent.remember(state_old, relative_action, reward, state_new, game_over)

        if game_over:
            # train long term memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > high_score:
                high_score = score

            print("Game", agent.n_games, "Score", score, "High Score:", high_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
