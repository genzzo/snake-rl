from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pygame
from pygame.event import Event

from .game import SnakeGameDirection


class ControllerInterface(ABC):
    @abstractmethod
    def get_direction(self, events: Optional[List[Event]] = None) -> SnakeGameDirection | None:
        """Return the next direction as (dx, dy) or None if no change."""
        pass


class KeyboardController(ControllerInterface):
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


class ReplayController(ControllerInterface):
    def __init__(self, moves: List[SnakeGameDirection]):
        """
        moves = list of direction tuples, e.g.:
        [(1,0), (1,0), (0,-1), ...]
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
