import random
from collections import deque
from typing import Tuple, Union, Literal, NamedTuple
from enum import Enum


class SnakeGameUpdateResult(Enum):
    GAME_OVER = 0
    ATE_FOOD = 1
    MOVED = 2


class SnakeGameDirection(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class Point(NamedTuple):
    x: int
    y: int


class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.available_positions = {
            Point(x, y) for x in range(self.width) for y in range(self.height)
        }
        self._spawn_food()
        self.snake = deque([random.choice(list(self.available_positions))])
        self.direction = (
            SnakeGameDirection.DOWN
            if self.snake[0][1] < self.height // 2
            else SnakeGameDirection.UP
        )
        self.available_positions.remove(self.snake[0])
        self.game_over = False
        self.game_over_reason = None

    def _get_movement_tuple_from_direction(
        self, direction: SnakeGameDirection
    ) -> Tuple[int, int]:
        switcher = {
            SnakeGameDirection.UP: (0, -1),
            SnakeGameDirection.RIGHT: (1, 0),
            SnakeGameDirection.DOWN: (0, 1),
            SnakeGameDirection.LEFT: (-1, 0),
        }
        return switcher.get(direction, (0, 0))

    def _spawn_food(self):
        if self.available_positions:
            self.food = random.choice(list(self.available_positions))
            self.available_positions.remove(self.food)
        else:
            # Snake has filled the entire grid (game won)
            self.food = None

    def _checkCollision(
        self, head: Point
    ) -> Union[tuple[Literal[True], str], tuple[Literal[False], None]]:
        if not 0 <= head[0] < self.width or not 0 <= head[1] < self.height:
            return (True, "Snake hit wall")
        if head in self.snake:
            return (True, "Snake ate itself")

        return (False, None)

    def update(
        self, new_direction: SnakeGameDirection | None = None
    ) -> SnakeGameUpdateResult:
        """Game logic only, no controls."""
        if new_direction:
            # prevent reversing into itself
            new_x, new_y = self._get_movement_tuple_from_direction(new_direction)
            curr_x, curr_y = self._get_movement_tuple_from_direction(self.direction)
            if (new_x != -curr_x) or (new_y != -curr_y):
                self.direction = new_direction

        head_x, head_y = self.snake[0]
        dx, dy = self._get_movement_tuple_from_direction(self.direction)
        new_head = Point(head_x + dx, head_y + dy)

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
