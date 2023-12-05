import os
import sys
import time

import gymnasium as gym
import torch_ac

from . import utils
from .model import ACModel
from .ppo_algo import PPOAlgo
from .utils import device
from src.constants import GLOBAL_SEED
from src.env.feedback_env import FeedbackEnv


os.environ["PROJECT_STORAGE"] = os.path.join(os.getcwd(), "external_rl/storage")


def make_env(env_key, seed=None, render_mode=None, feedback_mode=None, max_steps=None):
    _env = gym.make(env_key, render_mode=render_mode)
    _env.reset(seed=seed)
    env = FeedbackEnv(_env, feedback_mode=feedback_mode, max_steps=max_steps)
    return env


class PPOAgent:
    """
    PPOAgent is a wrapper around the PPO algorithm implementation from
    https://github.com/lcswillems/rl-starter-files for MinGrid and BabyAI environments.

    Args:
        env_name (str): name of the environment (e.g. "BabyAI-GoToObj-v0")
        seeds (list): list of seeds for the environment
        medium (bool, optional): whether to use the medium size model
        feedback_mode (str, optional): feedback mode for the environment
        max_steps (int, optional): maximum number of steps per episode
    """

    def __init__(
        self, env_name, seeds, medium=True, feedback_mode=None, max_steps=None
    ):
        self.args = {
            "algo": "ppo",
            "env": env_name,
            "model": None,
            "seeds": seeds,
            "log_interval": 1,
            "save_interval": 10,
            "procs": len(seeds),
            "epochs": 4,
            "batch_size_ppo": 256,
            "frames_per_proc": max_steps,
            "discount": 0.99,
            "lr": 0.001,
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "optim_eps": 1e-8,
            "optim_alpha": 0.99,
            "clip_eps": 0.2,
            "recurrence": 4
            if env_name in ["BabyAI-OpenTwoDoors-v0", "BabyAI-OpenRedBlueDoors-v0"]
            else 1,
            "text": True,
            "argmax": False,
            "feedback_mode": feedback_mode,
            "max_steps": max_steps,
        }
        self.medium = medium
        self.args["mem"] = self.args["recurrence"] > 1
        self.env = make_env(
            self.args["env"],
            self.args["seeds"][0],
            feedback_mode=feedback_mode,
            max_steps=max_steps,
        )
        self.model_dir = self._get_model_dir()

    def _get_model_dir(self):
        """
        Returns the path to the directory where the model weights are saved.
        """
        default_model_name = f"{self.args['env']}_{self.args['algo']}"
        model_name = self.args["model"] or default_model_name
        return os.path.join("external_rl", utils.get_model_dir(model_name))

    def _get_model(self):
        """
        Returns the model instance for the env and trained weights.
        """
        if not os.path.exists(
            os.path.join(
                self.model_dir, f"status_{'medium' if self.medium else 'expert'}.pt"
            )
        ):
            self._train_agent()

        return utils.Agent(
            self.env.observation_space,
            self.env.action_space,
            self.model_dir,
            argmax=False,
            num_envs=1,
            use_memory=self.args["mem"],
            use_text=self.args["text"],
        )

    def _train_agent(self, callback):
        """
        Trains the agent for the specified number of frames.
        This corresponds to the train.py script in the original implementation.
        """
        # Load loggers and Tensorboard writer
        txt_logger = utils.get_txt_logger(self.model_dir)
        csv_file, csv_logger = utils.get_csv_logger(self.model_dir)

        # Log command and all script arguments
        txt_logger.info(f"{' '.join(sys.argv)}\n")
        txt_logger.info(f"{self.args}\n")

        # Set seed for all randomness sources
        utils.seed(GLOBAL_SEED)

        # Set device
        txt_logger.info(f"Device: {device}\n")

        # Load environments
        envs = []
        for s in self.args["seeds"]:
            envs.append(
                make_env(
                    self.args["env"],
                    s,
                    feedback_mode=self.args["feedback_mode"],
                    max_steps=self.args["max_steps"],
                )
            )
        txt_logger.info(
            f"Environments loaded (using feedback_mode = {self.args['feedback_mode']}, max_steps = {self.args['max_steps']})\n"
        )

        # Load training status
        try:
            status = utils.get_status(self.model_dir)
        except:
            status = {"num_frames": 0, "update": 0}
        txt_logger.info("Training status loaded\n")

        # Load observations preprocessor
        obs_space, preprocess_obss = utils.get_obss_preprocessor(
            envs[0].observation_space
        )
        if "vocab" in status:
            preprocess_obss.vocab.load_vocab(status["vocab"])
        txt_logger.info("Observations preprocessor loaded")

        # Load model
        acmodel = ACModel(
            obs_space, envs[0].action_space, self.args["mem"], self.args["text"]
        )
        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
        acmodel.to(device)
        txt_logger.info("Model loaded\n")
        txt_logger.info(f"{acmodel}\n")

        # Load algo
        algo = PPOAlgo(
            envs,
            acmodel,
            device,
            self.args["frames_per_proc"],
            self.args["discount"],
            self.args["lr"],
            self.args["gae_lambda"],
            self.args["entropy_coef"],
            self.args["value_loss_coef"],
            self.args["max_grad_norm"],
            self.args["recurrence"],
            self.args["optim_eps"],
            self.args["clip_eps"],
            self.args["epochs"],
            self.args["batch_size_ppo"],
            preprocess_obss,
        )

        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")

        # Train model
        num_frames = status["num_frames"]
        update = status["update"]
        start_time = time.time()

        should_stop = False
        while not should_stop:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            should_stop = callback(exps, logs, self.args["seeds"])

            # Print logs
            if update % self.args["log_interval"] == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(
                    logs["reshaped_return_per_episode"]
                )
                num_frames_per_episode = utils.synthesize(
                    logs["num_frames_per_episode"]
                )

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [
                    logs["entropy"],
                    logs["value"],
                    logs["policy_loss"],
                    logs["value_loss"],
                    logs["grad_norm"],
                ]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(
                        *data
                    )
                )

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

            # Save status
            if (
                self.args["save_interval"] > 0
                and update % self.args["save_interval"] == 0
            ):
                status = {
                    "num_frames": num_frames,
                    "update": update,
                    "model_state": acmodel.state_dict(),
                    "optimizer_state": algo.optimizer.state_dict(),
                }
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, self.model_dir)
                txt_logger.info("Status saved")
