import random
from collections import deque
from typing import Tuple, Union


class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = deque([(10, 10)])
        self.direction = (1, 0)
        self.available_positions = {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.snake
        }
        self.spawn_food()
        self.game_over = False

    def spawn_food(self):
        if self.available_positions:
            self.food = random.choice(list(self.available_positions))
            self.available_positions.remove(self.food)
        else:
            # Snake has filled the entire grid (game won)
            self.food = None

    def update(self, new_direction: Union[Tuple[int, int], None] = None):
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

        # collision with walls or itself
        if (
            not 0 <= new_head[0] < self.width
            or not 0 <= new_head[1] < self.height
            or new_head in self.snake
        ):
            self.game_over = True
            return

        self.snake.appendleft(new_head)
        self.available_positions.discard(new_head)

        if new_head == self.food:
            self.spawn_food()
        else:
            tail = self.snake.pop()
            self.available_positions.add(tail)
