import random
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pygame
import torch
from numpy.typing import NDArray
from pygame.event import Event

from .game import Point, SnakeGame, SnakeGameDirection
from .model import LinearQNet, QTrainer

type AgentState = NDArray[np.int_]


class RelativeAction(Enum):
    """Relative actions: straight, turn right, turn left."""

    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2


class KeyboardController:
    def __init__(self) -> None:
        self.pressed_keys: List[int] = []
        self.key_map: Dict[int, SnakeGameDirection] = {
            pygame.K_UP: SnakeGameDirection.UP,
            pygame.K_DOWN: SnakeGameDirection.DOWN,
            pygame.K_LEFT: SnakeGameDirection.LEFT,
            pygame.K_RIGHT: SnakeGameDirection.RIGHT,
        }

    def get_direction(self, events: Optional[List[Event]] = None) -> SnakeGameDirection | None:
        if events is None:
            events = []

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in self.key_map:
                    if event.key in self.pressed_keys:
                        self.pressed_keys.remove(event.key)
                    self.pressed_keys.append(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in self.key_map:
                    if event.key in self.pressed_keys:
                        self.pressed_keys.remove(event.key)

        if self.pressed_keys:
            return self.key_map[self.pressed_keys[-1]]
        return None


class AgentController:
    def __init__(
        self,
        max_memory: int = 100_000,
        batch_size: int = 1000,
        learning_rate: float = 0.001,
        load_from_checkpoint: bool = True,
        checkpoint_file: str = "checkpoint.pth",
    ) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory: deque[tuple[AgentState, RelativeAction, int, AgentState, bool]] = deque(
            maxlen=max_memory
        )
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=learning_rate, gamma=self.gamma)
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.plot_scores: list[float] = []
        self.plot_mean_scores: list[float] = []

        # Try to load from checkpoint if requested
        if load_from_checkpoint:
            try:
                metadata = self.trainer.load_checkpoint(self.checkpoint_file)
                self.n_games = metadata["n_games"]
                self.plot_scores = metadata.get("plot_scores", [])
                self.plot_mean_scores = metadata.get("plot_mean_scores", [])
                print(f"Loaded checkpoint: starting from game {self.n_games}")
            except FileNotFoundError:
                print(f"No checkpoint found at {checkpoint_file}, starting fresh training")

    def get_state(self, game: SnakeGame) -> AgentState:
        boundary_threshold = 20

        head = game.snake[0]
        point_l = Point(head.x - boundary_threshold, head.y)
        point_r = Point(head.x + boundary_threshold, head.y)
        point_u = Point(head.x, head.y - boundary_threshold)
        point_d = Point(head.x, head.y + boundary_threshold)

        dir_l = game.direction == SnakeGameDirection.LEFT
        dir_r = game.direction == SnakeGameDirection.RIGHT
        dir_u = game.direction == SnakeGameDirection.UP
        dir_d = game.direction == SnakeGameDirection.DOWN

        state = [
            # Danger straight
            (dir_r and game.check_collision(point_r)[0])
            or (dir_l and game.check_collision(point_l)[0])
            or (dir_u and game.check_collision(point_u)[0])
            or (dir_d and game.check_collision(point_d)[0]),
            # Danger right
            (dir_u and game.check_collision(point_r)[0])
            or (dir_d and game.check_collision(point_l)[0])
            or (dir_l and game.check_collision(point_u)[0])
            or (dir_r and game.check_collision(point_d)[0]),
            # Danger left
            (dir_d and game.check_collision(point_r)[0])
            or (dir_u and game.check_collision(point_l)[0])
            or (dir_r and game.check_collision(point_u)[0])
            or (dir_l and game.check_collision(point_d)[0]),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.snake[0].x,  # food left
            game.food.x > game.snake[0].x,  # food right
            game.food.y < game.snake[0].y,  # food up
            game.food.y > game.snake[0].y,  # food down
        ]

        return np.array(state, dtype=int)

    def _action_to_one_hot(self, action: RelativeAction) -> list[int]:
        """Convert RelativeAction enum to one-hot encoded vector."""
        one_hot = [0, 0, 0]
        one_hot[action.value] = 1
        return one_hot

    def remember(
        self,
        state: AgentState,
        action: RelativeAction,
        reward: int,
        next_state: AgentState,
        game_over: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self) -> None:
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        # Convert RelativeAction enums to one-hot encoded vectors
        actions_one_hot = [self._action_to_one_hot(action) for action in actions]
        self.trainer.train_step(states, actions_one_hot, rewards, next_states, game_overs)

    def train_short_memory(
        self,
        state: AgentState,
        action: RelativeAction,
        reward: int,
        next_state: AgentState,
        game_over: bool,
    ) -> None:
        # Convert RelativeAction enum to one-hot encoded vector
        action_one_hot = self._action_to_one_hot(action)
        self.trainer.train_step(state, action_one_hot, reward, next_state, game_over)

    def get_action(self, state: AgentState) -> RelativeAction:
        """Get relative action: straight, turn right, or turn left."""
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            # Random exploration: choose random relative action
            move = random.randint(0, 2)
            return RelativeAction(move)
        else:
            # Exploitation: use model prediction
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            return RelativeAction(move)

    def convert_action_to_direction(
        self, action: RelativeAction, current_direction: SnakeGameDirection
    ) -> SnakeGameDirection:
        """Convert relative action to absolute direction based on current direction."""
        if action == RelativeAction.STRAIGHT:
            return current_direction

        # Define clockwise direction order
        clockwise = [
            SnakeGameDirection.UP,
            SnakeGameDirection.RIGHT,
            SnakeGameDirection.DOWN,
            SnakeGameDirection.LEFT,
        ]
        current_idx = clockwise.index(current_direction)

        if action == RelativeAction.RIGHT:
            return clockwise[(current_idx + 1) % 4]
        else:
            return clockwise[(current_idx - 1) % 4]

    def get_direction(self, events: Optional[List[Event]] = None) -> SnakeGameDirection | None:
        """Not used for AgentController."""
        return None

    def save_checkpoint(self, high_score: int = 0, **kwargs: Any) -> None:
        """Save training checkpoint.

        Args:
            high_score: The highest score achieved so far
            **kwargs: Additional metadata to save in checkpoint
        """
        self.trainer.save_checkpoint(
            file_name=self.checkpoint_file,
            n_games=self.n_games,
            high_score=high_score,
            plot_scores=self.plot_scores,
            plot_mean_scores=self.plot_mean_scores,
            **kwargs,
        )


class ReplayController:
    def __init__(self, moves: List[SnakeGameDirection]):
        """
        Initialize replay controller with a list of SnakeGameDirection moves.

        Args:
            moves: List of SnakeGameDirection enums from a recorded replay
        """
        self.moves = moves
        self.index = 0

    def get_direction(self, events: Optional[List[Event]] = None) -> SnakeGameDirection | None:
        if self.index < len(self.moves):
            direction = self.moves[self.index]
            self.index += 1
            return direction
        else:
            # If replay is done, keep going straight
            return None

    @classmethod
    def from_replay_file(cls, file_path: str) -> "ReplayController":
        """Create a ReplayController from a saved replay file.

        Args:
            file_path: Path to the replay JSON file

        Returns:
            ReplayController instance loaded with the replay data
        """
        from .replay import ReplayRecorder

        _, frames = ReplayRecorder.load(file_path)
        moves = [SnakeGameDirection(frame.direction) for frame in frames]
        return cls(moves)
