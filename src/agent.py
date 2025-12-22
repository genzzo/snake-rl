from collections import deque
from typing import Any

from src.game import SnakeGame, SnakeGameDirection

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamme = 0  # discount rate
        self.memory: deque[int] = deque(maxlen=MAX_MEMORY)

    def get_state(self, game: SnakeGame):
        pass

    def remember(
        self,
        state: Any,
        action: SnakeGameDirection | None,
        reward: int,
        next_state: Any,
        game_over: bool,
    ):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(
        self,
        state: Any,
        action: SnakeGameDirection | None,
        reward: int,
        next_state: Any,
        game_over: bool,
    ):
        pass

    def get_action(self, state: Any) -> SnakeGameDirection | None:
        pass


def train_agent(game: SnakeGame):
    # plot_scores: list[float] = []
    # plot_mean_scores: list[float] = []
    # total_score = 0
    high_score = 0

    agent = Agent()

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        game.update(final_move)
        reward, game_over, score = (0, game.game_over, len(game.snake))

        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train long term memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > high_score:
                high_score = score
