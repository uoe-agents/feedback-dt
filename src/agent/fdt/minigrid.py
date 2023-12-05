import torch.nn as nn

from .base import FDTAgent


class MinigridFDTAgent(FDTAgent):
    def __init__(self, *args, use_rgb=True, **kwargs):
        self.use_rgb = use_rgb
        print(f"Using rgb: {self.use_rgb}")
        super().__init__(*args, **kwargs)

    def create_state_embedding_model(self):
        """Create the state embedding model."""
        if not self.use_rgb:
            print("Creating state embedding model for symbolic observations")
            # minimal CNN for embedding symbolic observations
            self.state_embedding_model = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(512, self.hidden_size),
                nn.Tanh(),
            )
            return

        # CNN for embedding full image observations
        print("Creating state embedding model for rgb image observations")
        self.state_embedding_model = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(40000, self.hidden_size),
            nn.Tanh(),
        )
