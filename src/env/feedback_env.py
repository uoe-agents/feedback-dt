from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces.discrete import Discrete

from src.dataset.custom_feedback_verifier import RuleFeedback
from src.dataset.custom_feedback_verifier import TaskFeedback


class FeedbackEnv:
    """
    FeedbackEnv is a wrapper around a gymnasium environment that adds the ability to provide feedback to the agent.

    Args:
        env (gym.Env): The environment to wrap.
        feedback_mode (str): The type of feedback to provide to the agent. Can be one of "rule", "task", "numerical", or "mixed".
        max_steps (int): The maximum number of steps to take in the environment. If None, then the max_steps of the wrapped environment is used.
    """

    def __init__(
        self, env: gym.Env, feedback_mode: Optional[str], max_steps: Optional[int]
    ) -> None:
        self.env = env
        self.feedback_mode = feedback_mode
        self._max_steps = max_steps
        self.steps_taken = 0
        if self.feedback_mode:
            self.rule_fv = RuleFeedback()
            self.task_fv = TaskFeedback(self.env)

    def get_base_env(self):
        """Return the base gym environment."""
        return self.env

    def rule_feedback(self, action):
        """
        Return rule feedback for the given action.

        Args:
            action (int): The action taken by the agent.

        Returns:
            str or None: The rule feedback for the given action.
        """
        return (
            self.rule_fv.verify_feedback(self.env, action)
            if self.feedback_mode != "task"
            else None
        )

    def task_feedback(self, action):
        """
        Return task feedback for the given action.

        Args:
            action (int): The action taken by the agent.

        Returns:
            str or None: The rule feedback for the given action.
        """
        return (
            self.task_fv.verify_feedback(self.env, action)
            if self.feedback_mode != "rule"
            else None
        )

    def get_feedback(self, rule_feedback, task_feedback):
        """Return feedback depending on the specified feedback mode."""
        if self.feedback_mode == "rule":
            return rule_feedback
        if self.feedback_mode == "task":
            return task_feedback
        if self.feedback_mode == "numerical":
            if task_feedback != "No feedback available.":
                return "1"
            if rule_feedback != "No feedback available.":
                return "-1"
            return "0"
        else:
            if rule_feedback == "No feedback available.":
                return task_feedback
            return rule_feedback

    def step(self, action):
        """
        Execute the given action in the environment and return the resulting observation, reward, terminated, truncated, and feedback.

        Args:
            action (int): The action to take in the environment.

        Returns:
            tuple: The observation, reward, terminated, truncated, and feedback.
        """
        if not self.feedback_mode:
            obs, reward, terminated, truncated, _ = self.env.step(action)
            return obs, reward, terminated, truncated, None

        # get rule feedback (before taking action)
        rule_feedback = self.rule_feedback(action)

        # call env.step
        obs, reward, terminated, truncated, _ = self.env.step(action)

        # check if max steps reached
        self.steps_taken += 1
        if (
            self.max_steps is not None
            and self.steps_taken >= self.max_steps
            and not terminated
        ):
            truncated = True
        else:
            truncated = False

        # get task feedback (after taking action)
        task_feedback = self.task_feedback(action)

        # get feedback + return tuple
        feedback = self.get_feedback(rule_feedback, task_feedback)
        return obs, reward, terminated, truncated, feedback

    def reset(self, *args, **kwargs):
        """Reset the environment."""
        self.steps_taken = 0
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        """Render the environment."""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Close the environment."""
        return self.env.close()

    def get_frame(self, *args, **kwargs):
        """Get the current frame of the environment."""
        return self.env.get_frame(*args, **kwargs)

    def get_mission(self):
        """Return the current mission string."""
        return self.env.instrs.surface(self.env)

    def room_from_pos(self, *args, **kwargs):
        """Return the room from the given position."""
        return self.env.room_from_pos(*args, **kwargs)

    @property
    def action_space(self):
        # Excluding the "Done" action
        return Discrete(6)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def actions(self):
        return self.env.actions

    @property
    def agent_pos(self):
        return self.env.agent_pos

    @property
    def front_pos(self):
        return self.env.front_pos

    @property
    def carrying(self):
        return self.env.carrying

    @property
    def grid(self):
        return self.env.grid

    @property
    def mission(self):
        return self.env.mission

    @property
    def max_steps(self):
        tmp = self._max_steps or np.inf
        return min(tmp, self.env.max_steps)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def instrs(self):
        return self.env.instrs

    @property
    def step_count(self):
        return self.env.step_count
