import itertools
import logging
import os
import sys
from datetime import datetime

import gymnasium as gym
import pygame
from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl
from PIL import Image

from src.env.feedback_env import FeedbackEnv

ACTION_TO_STR = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
}


class DemoManualControl(ManualControl):
    """
    Class for manually controlling an environment for demonstration purposes.

    Args:
        config (str): Name of the environment config to load.
        seed (int, optional): Seed for the environment.
        record (bool, optional,): Whether to record the episode.
        speed (int, optional): Speed of the replay.
        save_log (bool, optional): Whether to save logs.
        reset_env (bool, optional): Whether to allow resetting by pressing 'r'.
    """

    def __init__(
        self,
        config: str,
        seed=None,
        record=False,
        speed=1,
        save_log=False,
        reset_env=False,
    ) -> None:
        self.config = config
        self.seed = seed
        self.env = self._make_env(config, render_mode="human")
        super().__init__(self.env, seed)
        self.record = record
        self.speed = speed
        self.reset_env = reset_env
        self.save_dir = self._get_save_dir()
        self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if save_log:
            logging.basicConfig(
                filename=f"{self.save_dir}/{self.date}_{self.config.split('-')[1]}.log",
                level=logging.INFO,
            )
        self.actions = []
        self.frames = []

    def _get_save_dir(self):
        """Get the directory to save demos to."""
        project_dir = os.path.abspath(os.getcwd())
        save_dir = os.path.join(project_dir, "demos")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        else:
            print(f"Path {save_dir} exists")
        return save_dir

    def _make_env(self, level, render_mode):
        """Make an environment."""
        return gym.make(level, render_mode=render_mode)

    def reset(self, seed=None):
        """Reset the environment and recording state."""
        self.actions = []
        self.frames = []
        self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.env.reset(seed=seed)
        mission = self.env.mission
        self.env = FeedbackEnv(
            self.env, feedback_mode="all", max_steps=self.env.max_steps
        )
        message = f"step=0, mission='{mission}'"
        logging.info(message)
        self.env.render()

    def start(self):
        """
        Start the window display with blocking event loop.

        Overriding the start method from minigrid.manual_control.ManualControl.
        This is to allow quitting with closing the window and executing code outside of start().

        """
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._close_window()
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    if event.key == "escape":
                        self._close_window()
                        break
                    print(event.key)
                    if event.key == "r":
                        if self.reset_env:
                            if self.record:
                                self.replay_episode()
                            self.reset(self.seed)
                        else:
                            self._close_window()
                    self.key_handler(event)

    def _close_window(self):
        if self.record:
            self.replay_episode()
        self.env.close()
        pygame.quit()
        sys.exit()

    def step(self, action: Actions):
        """Take a step in the environment."""
        _, reward, terminated, truncated, feedback = self.env.step(action)
        self.actions.append(action)
        message = f"step={self.env.step_count}, action={ACTION_TO_STR[action]}, feedback={feedback}, reward={reward:.2f}"

        if not (terminated or truncated):
            logging.info(message)
            self.env.render()
        else:
            message += ", terminated" if terminated else ", truncated"
            logging.info(message)
            if self.reset_env:
                if self.record:
                    self.replay_episode()
                self.reset(self.seed)
            else:
                self._close_window()

    def replay_episode(self):
        """Replay the episode and save the frames."""
        rgb_env = self._make_env(self.config, render_mode="rgb_array")
        rgb_env.reset(seed=self.seed)
        self.frames.append(Image.fromarray(rgb_env.render()))
        for action in self.actions:
            rgb_env.step(action)
            self.frames.append(Image.fromarray(rgb_env.render()))
        self.save_images()
        self.save_gif()

    def save_images(self):
        """Save the episode frames as images."""
        for i, frame in enumerate(self.frames):
            save_path = (
                f"{self.save_dir}/{self.date}_{self.config.split('-')[1]}_step-{str(i)}"
            )
            frame.save(save_path + ".pdf")
            frame.save(save_path + ".png")

    def save_gif(self):
        """Save the episode frames as a gif."""
        save_path = f"{self.save_dir}/{self.date}_{self.config.split('-')[1]}.gif"
        frames_extended = list(
            itertools.chain.from_iterable(
                itertools.repeat(f, int(10 / self.speed)) for f in self.frames
            )
        )
        frames_extended[0].save(
            save_path,
            save_all=True,
            append_images=frames_extended[1:],
            duration=len(frames_extended),
            loop=0,
        )
