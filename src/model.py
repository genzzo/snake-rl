import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()  # type: ignore
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = "model.pth") -> None:
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name: str = "model.pth") -> None:
        """Load model weights from a file."""
        model_folder_path = "./model"
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            self.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {file_path}")


class QTrainer:
    def __init__(self, model: LinearQNet, learning_rate: float, gamma: float) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        done: Any,
    ) -> None:
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # reshape for a single sample
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()  # type: ignore

    def save_checkpoint(
        self,
        file_name: str = "checkpoint.pth",
        n_games: int = 0,
        high_score: int = 0,
        **kwargs: Any,
    ) -> None:
        """Save complete training checkpoint including model, optimizer state, and metadata."""
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "n_games": n_games,
            "high_score": high_score,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            **kwargs,
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved: {file_path}")

    def load_checkpoint(self, file_name: str = "checkpoint.pth") -> dict[str, Any]:
        """Load complete training checkpoint.

        Returns:
            Dictionary containing loaded metadata (n_games, high_score, etc.)
        """
        model_folder_path = "./model"
        file_path = os.path.join(model_folder_path, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Extract metadata
        metadata = {
            "n_games": checkpoint.get("n_games", 0),
            "high_score": checkpoint.get("high_score", 0),
            "learning_rate": checkpoint.get("learning_rate", self.learning_rate),
            "gamma": checkpoint.get("gamma", self.gamma),
        }

        # Include any additional saved data
        for key, value in checkpoint.items():
            excluded_keys = [
                "model_state_dict",
                "optimizer_state_dict",
                "n_games",
                "high_score",
                "learning_rate",
                "gamma",
            ]
            if key not in excluded_keys:
                metadata[key] = value

        print(f"Checkpoint loaded: {file_path}")
        print(f"Resuming from game {metadata['n_games']} with high score {metadata['high_score']}")

        return metadata
