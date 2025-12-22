"""Replay recording and playback functionality for Snake RL."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

from .game import SnakeGameDirection


@dataclass
class ReplayFrame:
    """A single frame in a replay."""

    direction: int  # SnakeGameDirection value
    snake_positions: List[tuple[int, int]]
    food_position: tuple[int, int]
    score: int


class ReplayRecorder:
    """Records gameplay for later playback."""

    def __init__(self) -> None:
        self.frames: List[ReplayFrame] = []
        self.metadata: Dict[str, Any] = {}
        self.is_recording = False

    def start_recording(self, **metadata: Any) -> None:
        """Start recording a new replay.

        Args:
            **metadata: Additional metadata to store (e.g., game_mode, initial_seed)
        """
        self.frames = []
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            **metadata,
        }
        self.is_recording = True

    def record_frame(
        self,
        direction: SnakeGameDirection,
        snake_positions: List[tuple[int, int]],
        food_position: tuple[int, int],
        score: int,
    ) -> None:
        """Record a single frame of gameplay."""
        if not self.is_recording:
            return

        frame = ReplayFrame(
            direction=direction.value,
            snake_positions=snake_positions,
            food_position=food_position,
            score=score,
        )
        self.frames.append(frame)

    def stop_recording(self) -> None:
        """Stop recording."""
        self.is_recording = False

    def save(self, file_name: str | None = None, folder: str = "./replays") -> str:
        """Save the recorded replay to a file.

        Args:
            file_name: Name of the file (generated if None)
            folder: Folder to save replays in

        Returns:
            Path to the saved replay file
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            score = self.metadata.get("final_score", 0)
            file_name = f"replay_{timestamp}_score{score}.json"

        file_path = os.path.join(folder, file_name)

        replay_data: dict[str, Any] = {
            "metadata": self.metadata,
            "frames": [asdict(frame) for frame in self.frames],
        }

        with open(file_path, "w") as f:
            json.dump(replay_data, f, indent=2)

        print(f"Replay saved: {file_path}")
        return file_path

    @staticmethod
    def load(file_path: str) -> tuple[Dict[str, Any], List[ReplayFrame]]:
        """Load a replay from a file.

        Args:
            file_path: Path to the replay file

        Returns:
            Tuple of (metadata, frames)
        """
        with open(file_path) as f:
            data = json.load(f)

        metadata = data["metadata"]
        frames = [
            ReplayFrame(
                direction=frame["direction"],
                snake_positions=[tuple(pos) for pos in frame["snake_positions"]],
                food_position=tuple(frame["food_position"]),
                score=frame["score"],
            )
            for frame in data["frames"]
        ]

        return metadata, frames

    @staticmethod
    def list_replays(folder: str = "./replays") -> List[str]:
        """List all available replay files in the folder.

        Args:
            folder: Folder containing replays

        Returns:
            List of replay file paths
        """
        if not os.path.exists(folder):
            return []

        replay_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".json") and f.startswith("replay_")
        ]
        return sorted(replay_files, reverse=True)  # Most recent first
