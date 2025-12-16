import random
from collections import deque
from typing import Tuple, Union, Literal
from enum import Enum


class SnakeGameUpdateResult(Enum):
    GAME_OVER = 0
    ATE_FOOD = 1
    MOVED = 2


class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.available_positions = {
            (x, y) for x in range(self.width) for y in range(self.height)
        }
        self._spawn_food()
        self.snake = deque([random.choice(list(self.available_positions))])
        self.direction = (0, 1) if self.snake[0][1] < self.height // 2 else (0, -1)
        self.available_positions.remove(self.snake[0])
        self.game_over = False
        self.game_over_reason = None

    def _spawn_food(self):
        if self.available_positions:
            self.food = random.choice(list(self.available_positions))
            self.available_positions.remove(self.food)
        else:
            # Snake has filled the entire grid (game won)
            self.food = None

    def _checkCollision(
        self, head_position: tuple[int, int]
    ) -> Union[tuple[Literal[True], str], tuple[Literal[False], None]]:
        if (
            not 0 <= head_position[0] < self.width
            or not 0 <= head_position[1] < self.height
        ):
            return (True, "Snake hit wall")
        if head_position in self.snake:
            return (True, "Snake ate itself")

        return (False, None)

    def update(
        self, new_direction: Union[Tuple[int, int], None] = None
    ) -> SnakeGameUpdateResult:
        """Game logic only, no controls."""
        if new_direction:
            # prevent reversing into itself
            if (
                new_direction[0] != -self.direction[0]
                or new_direction[1] != -self.direction[1]
            ):
                self.direction = new_direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        (isColliding, collisionReason) = self._checkCollision(new_head)
        if isColliding:
            self.game_over = True
            self.game_over_reason = collisionReason
            return SnakeGameUpdateResult.GAME_OVER

        self.snake.appendleft(new_head)
        self.available_positions.discard(new_head)

        if new_head == self.food:
            self._spawn_food()
            return SnakeGameUpdateResult.ATE_FOOD
        else:
            tail = self.snake.pop()
            self.available_positions.add(tail)
            return SnakeGameUpdateResult.MOVED
