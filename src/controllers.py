import pygame
from pygame.event import Event
from typing import List, Tuple, Literal, Optional, Dict
from abc import ABC, abstractmethod


class ControllerInterface(ABC):
    @abstractmethod
    def get_direction(
        self, events: Optional[List[Event]] = None
    ) -> Tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]] | None:
        """Return the next direction as (dx, dy) or None if no change."""
        pass


class KeyboardController(ControllerInterface):
    def __init__(self):
        self.pressed_keys: List[int] = []
        self.key_map: Dict[int, Tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]] = {
            pygame.K_UP: (0, -1),
            pygame.K_DOWN: (0, 1),
            pygame.K_LEFT: (-1, 0),
            pygame.K_RIGHT: (1, 0),
        }

    def get_direction(
        self, events: Optional[List[Event]] = None
    ) -> Tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]] | None:
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
    def __init__(self, moves: List[Tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]]):
        """
        moves = list of direction tuples, e.g.:
        [(1,0), (1,0), (0,-1), ...]
        """
        self.moves = moves
        self.index = 0

    def get_direction(
        self, events: Optional[List[Event]] = None
    ) -> Tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]] | None:
        if self.index < len(self.moves):
            direction = self.moves[self.index]
            self.index += 1
            return direction
        else:
            # If replay is done, keep going straight
            return None
