import pygame
from typing import List, Tuple
from abc import ABC, abstractmethod


class ControllerInterface(ABC):
    @abstractmethod
    def get_direction(self) -> Tuple[int, int] | None:
        pass


class KeyboardController(ControllerInterface):
    def __init__(self):
        self.dir = None

    def get_direction(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.dir = (0, -1)
        elif keys[pygame.K_DOWN]:
            self.dir = (0, 1)
        elif keys[pygame.K_LEFT]:
            self.dir = (-1, 0)
        elif keys[pygame.K_RIGHT]:
            self.dir = (1, 0)

        return self.dir


class ReplayController(ControllerInterface):
    def __init__(self, moves: List[Tuple[int, int]]):
        """
        moves = list of direction tuples, e.g.:
        [(1,0), (1,0), (0,-1), ...]
        """
        self.moves = moves
        self.index = 0

    def get_direction(self):
        if self.index < len(self.moves):
            direction = self.moves[self.index]
            self.index += 1
            return direction
        else:
            # If replay is done, keep going straight
            return None
