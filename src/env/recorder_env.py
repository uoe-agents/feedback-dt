import os
import shutil
from typing import Optional

import cv2
import gymnasium as gym

from .feedback_env import FeedbackEnv


class RecorderEnv(FeedbackEnv):
    """
    RecorderEnv is a wrapper around a gymnasium environment that records the environment as a video.

    Args:
        env (gym.Env): The environment to wrap.
        feedback_mode (str): The type of feedback to provide to the agent. Can be one of "rule", "task", "numerical", or "mixed".
        directory (str): The directory to save the video.
        filename (str): The name of the video file.
        auto_release (bool): Whether to automatically release the video when the episode is done.
        size (tuple[int, int]): The size of the video.
        fps (int): The FPS of the video.
        rgb (bool): Whether to save the video as RGB or BGR.
        max_steps (int): The maximum number of steps to take in the environment. If None, then the max_steps of the wrapped environment is used.
    """

    def __init__(
        self,
        env: gym.Env,
        feedback_mode: Optional[str],
        directory,
        filename,
        auto_release=True,
        size=None,
        fps=30,
        rgb=True,
        max_steps=None,
    ):
        super().__init__(env, feedback_mode, max_steps)
        self.directory = os.path.join(directory, "recordings")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.path = os.path.join(self.directory, f"{filename}.mp4")
        self.auto_release = auto_release
        self.size = size
        self.active = True
        self.fps = fps
        self.rgb = rgb

        if self.size is None:
            self.reset()
            self.size = self.render().shape[:2][::-1]

    def pause(self):
        """Pause the recording."""
        self.active = False

    def resume(self):
        """Resume the recording."""
        self.active = True

    def _start(self):
        """Start the video writer."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)

    def _write(self, obs=None):
        """Write a frame to the video file."""
        if not self.active:
            return
        frame = self.render()
        self._writer.write(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.rgb else frame
        )

    def release(self):
        """Release the video writer."""
        self._writer.release()

    def reset(self, *args, **kwargs):
        """Reset the environment."""
        obs, info = super().reset(*args, **kwargs)
        self._start()
        self._write(obs)
        return obs, info

    def step(self, *args, **kwargs):
        """Take a step in the environment."""
        data = super().step(*args, **kwargs)
        self._write(data[0])
        if self.auto_release and data[2]:
            self.release()
        return data

    def save_as(self, label):
        """Save the video to the given filename."""
        shutil.copy(self.path, os.path.join(self.directory, f"{label}.mp4"))
