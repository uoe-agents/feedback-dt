import os
from dataclasses import dataclass
from typing import Any
from typing import Dict

import numpy as np
import torch
from torch import nn


@dataclass
class AgentInput:
    mission_embeddings: Any
    states: Any
    actions: Any
    rewards: Any
    returns_to_go: Any
    timesteps: Any
    feedback_embeddings: Any
    attention_mask: Any


class Agent(nn.Module):
    """
    Agent is the base class used to represent any offline-rl-ish trainable agent.
    """

    def __init__(self) -> None:
        super().__init__()

    def _forward(self, input: AgentInput) -> Any:
        """Operate on a given input and return the result. Override in subclasses."""
        pass

    def _compute_loss(self, input: AgentInput, output: Any) -> float:
        """Compute the loss given an input and output. Override in subclasses."""
        pass

    def forward(self, input: AgentInput, **kwargs) -> Dict:
        """Perform a forward pass given an input, and return the resulting loss. Override in subclasses."""
        output = self._forward(input)
        loss = self._compute_loss(input, output)
        return {"loss": loss}

    def get_action(self, input: AgentInput, context=1, one_hot=False):
        """Sample an action from the model given an input. Override in subclasses."""
        pass

    def save_checkpoint(self, expt_dir, step):
        """
        Save a checkpoint of the model to the given directory.

        Args:
            expt_dir: The experiment directory to save the checkpoint to.
            step: The step number to use in the checkpoint filename.
        """
        dir = f"{expt_dir}/checkpoints"
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.state_dict(), f"{dir}/step_{step}.pt")

    def load_checkpoint(self, expt_dir, step):
        """
        Load a checkpoint of the model from the given directory.

        Args:
            expt_dir: The experiment directory to load the checkpoint from.
            step: The step number to use in the checkpoint filename.
        """
        print(f"Loading checkpoint {step}")
        dir = f"{expt_dir}/checkpoints"
        self.load_state_dict(torch.load(f"{dir}/step_{step}.pt"))


class RandomAgent(Agent):
    def __init__(self, act_dim) -> None:
        self.act_dim = act_dim
        super().__init__()

    def _forward(self, input: AgentInput) -> Any:
        return None

    def _compute_loss(self, input: AgentInput, output: Any) -> float:
        return torch.tensor(0.0, requires_grad=True)

    def get_action(self, input: AgentInput, context=1, one_hot=False):
        """Sample a random action from the model's action space."""
        return torch.tensor(
            np.random.random(self.act_dim).astype(np.float32),
            device=input.states.device,
        )
