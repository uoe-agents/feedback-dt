import os
import shutil

import gymnasium as gym
import numpy as np
from dopamine.replay_memory import circular_replay_buffer
from jsonc_parser.parser import JsoncParser
from tqdm import tqdm

from src.dataset.minari_dataset import MinariDataset
from src.dataset.minari_storage import name_dataset
from src.dataset.seeds import LEVELS_CONFIGS
from src.dataset.seeds import SeedFinder
from src.env.feedback_env import FeedbackEnv
from src.ppo.ppo_agent import PPOAgent
from src.utils.utils import discounted_cumsum
from src.utils.utils import get_minigrid_obs
from src.utils.utils import log
from src.utils.utils import normalise
from src.utils.utils import to_one_hot


class CustomDataset:
    """
    Class for generating a custom dataset for a given environment, seed and policy.

    Args:
        args (dict): A dictionary of user-specified arguments.
    """

    def __init__(self, args):
        self.args = args
        self.shard = None
        self.buffers = []
        self.steps = []
        self.ep_counts = []
        self.total_steps = 0
        self.total_episodes = 0
        self.seed_finder = SeedFinder()
        self.level = self.args["level"]
        self.num_train_seeds = self.args["num_train_seeds"]
        self.eps_per_seed = self.args["eps_per_seed"]
        self.category = self._get_category()
        self.train_config, self.test_configs = self._get_configs()
        self.train_seeds = self._get_train_seeds()
        self.max_steps = self._get_level_max_steps()
        self._determine_eps_per_shard()
        self.shard_list = []
        self.env = None
        self._ppo_best_return = -np.inf
        self._ppo_early_stop_count = 0

    def _get_configs(self):
        """
        Get the configs for the given level that are suitable for training.

        Check, for each possible config for the level, if the number of safe train seeds is non-zero and include only those.

        Returns:
            list: the configs.
        """
        test_configs = []
        for config in LEVELS_CONFIGS["original_tasks"][self.args["level"]]:
            seed_log = self.seed_finder.load_seeds(self.args["level"], config)
            if seed_log["n_train_seeds"]:
                train_config = config
            test_configs.append(config)
        return train_config, test_configs

    def _get_dataset(self):
        """
        Get a MinariDataset object, either by loading an existing dataset from local storage
        or by generating a new dataset.

        Returns:
            MinariDataset: the dataset object that was retrieved from storage or created.
        """
        dataset_name = name_dataset(self.args)
        minari_fp = os.environ.get("MINARI_DATASETS_PATH") or os.path.join(
            os.path.expanduser("~"), ".minari", "datasets"
        )
        self.fp, self.num_shards = os.path.join(minari_fp, dataset_name), 0

        if self.args["load_existing_dataset"] and os.path.exists(self.fp):
            print(f"Loading existing dataset {dataset_name}")
            self._load_dataset(self.fp)
        else:
            print(f"Creating dataset {dataset_name}")
            self._initialise_new_dataset()
            if self.args["policy"] == "random":
                self._generate_new_dataset()
            else:
                self._from_ppo_training()
            self.buffers = []
            self.steps = []
            self.ep_counts = []

        return self

    def _load_dataset(self, datset_dir):
        self.num_shards = 0
        for _ in os.listdir(datset_dir):
            self.num_shards += 1
        self.get_shard_list()

    def _get_category(self):
        """
        Get the category from the level.

        Returns:
            str: the category.
        """
        metadata_path = os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        metadata = JsoncParser.parse_file(metadata_path)["levels"]
        for level_group, levels in metadata.items():
            if self.args["level"] in levels:
                return level_group

    def _get_train_seeds(self):
        # choose random subset of train seeds for the train config
        seed_log = self.seed_finder.load_seeds(self.level, self.train_config)
        train_seeds = self.seed_finder.get_train_seeds(
            seed_log, self.level, self.num_train_seeds
        )
        return [
            int(s) for s in np.random.choice(train_seeds, size=self.num_train_seeds)
        ]

    def _get_level_max_steps(self):
        """
        Get the max steps for the environment.

        Returns:
            int: the max steps.
        """
        metadata_path = os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        metadata = JsoncParser.parse_file(metadata_path)
        level_metadata = metadata["levels"][self.category][self.args["level"]]

        seq_instrs_factor = 4 if level_metadata["mission_space"]["sequence"] else 1
        putnext_instrs_factor = 2 if level_metadata["putnext"] else 1
        max_instrs_factor = 1 * seq_instrs_factor * putnext_instrs_factor
        step_ceiling = 8**2 * 3**2 * 2

        try:
            max_steps = level_metadata[self.train_config]["max_steps"]
        except KeyError:
            max_steps = 0
        try:
            room_size = level_metadata[self.train_config]["room_size"]
        except KeyError:
            room_size = metadata["defaults"]["room"]["room_size"]
        try:
            num_rows = level_metadata[self.train_config]["num_rows"]
        except KeyError:
            num_rows = (
                metadata["defaults"]["maze"]["num_rows"]
                if level_metadata["maze"]
                else 1
            )
        try:
            num_cols = level_metadata[self.train_config]["num_cols"]
        except KeyError:
            num_cols = (
                metadata["defaults"]["maze"]["num_cols"]
                if level_metadata["maze"]
                else 1
            )
            tmp_max_steps = room_size**2 * num_rows * num_cols * max_instrs_factor
            global_max_steps = max(tmp_max_steps, max_steps)
        return min(global_max_steps, step_ceiling)

    def _determine_eps_per_shard(self):
        eps_per_shard = 128 if self.max_steps < 128 else self.args["eps_per_shard"]
        self.eps_per_shard = eps_per_shard

    def get_feedback_constant(self):
        """
        Get the constant feedback string depending on the feedback mode.

        Returns:
            str: the constant feedback string.
        """
        if self.args["feedback_mode"] == "numerical":
            return "0"
        return "No feedback available."

    def _initialise_buffers(self, num_buffers, obs_shape):
        """
        Initialise a set of buffers to store replay data.

        Args:
            num_buffers (int): The number of buffers to initialise.
            obs_shape (tuple): The shape of the observations.
        """
        self._determine_eps_per_shard()

        log(
            f"initialising {num_buffers} buffers of size {self.eps_per_shard}",
            with_tqdm=True,
        )
        for _ in range(num_buffers):
            self.buffers.append(self._create_buffer(obs_shape))
            self.steps.append(0)
            self.ep_counts.append(0)

    def _create_buffer(self, obs_shape):
        """
        Create a buffer to store replay data.

        Args:
            obs_shape (tuple): The shape of the observations.
        """
        num_eps = self.eps_per_shard

        log(
            f"creating buffer of size {num_eps * (self.max_steps + 1)} steps",
            with_tqdm=True,
        )

        return {
            "seeds": np.array([[0]] * ((self.max_steps + 1) * num_eps)),
            "missions": ["No mission available."] * ((self.max_steps + 1) * num_eps),
            "observations": np.array(
                [np.zeros(obs_shape)] * ((self.max_steps + 1) * num_eps),
                dtype=np.uint8,
            ),
            "actions": np.array(
                [[0]] * ((self.max_steps + 1) * num_eps),
                dtype=np.float32,
            ),
            "rewards": np.array(
                [[0]] * ((self.max_steps + 1) * num_eps),
                dtype=np.float32,
            ),
            "feedback": [self.get_feedback_constant()]
            * ((self.max_steps + 1) * num_eps),
            "terminations": np.array(
                [[0]] * ((self.max_steps + 1) * num_eps), dtype=bool
            ),
            "truncations": np.array(
                [[0]] * ((self.max_steps + 1) * num_eps), dtype=bool
            ),
        }

    def _flush_buffer(self, buffer_idx, obs_shape):
        """
        Flush a buffer to file and re-initialise it.

        Args:
            buffer_idx (int): The index of the buffer to flush.
            obs_shape (tuple): The shape of the observations.
        """

        # if buffer exists and isn't empty, first save it to file
        if (
            len(self.buffers) > buffer_idx
            and len(self.buffers[buffer_idx]["observations"]) > 0
            and np.any(self.buffers[buffer_idx]["observations"])  # check any nonzero
        ):
            self._save_buffer_to_minari_file(buffer_idx)

        if obs_shape is None:
            obs_shape = self.buffers[buffer_idx]["observations"][0].shape

        self.buffers[buffer_idx] = self._create_buffer(obs_shape)
        self.steps[buffer_idx] = 0
        self.ep_counts[buffer_idx] = 0

    def _save_buffer_to_minari_file(self, buffer_idx):
        """
        Save the data in a buffer to a MinariDataset file.

        Args:
            buffer_idx (int): The index of the buffer to save.
        """
        for key in self.buffers[buffer_idx].keys():
            self.buffers[buffer_idx][key] = self.buffers[buffer_idx][key][
                : self.steps[buffer_idx] + 1
            ]

        episode_terminals = (
            self.buffers[buffer_idx]["terminations"]
            + self.buffers[buffer_idx]["truncations"]
            if self.args["include_timeout"]
            else None
        )

        md = MinariDataset(
            level_group=self.category,
            level_name=self.args["level"],
            train_config=self.train_config,
            dataset_name=name_dataset(self.args),
            policy=self.args["policy"],
            feedback_mode=self.args["feedback_mode"],
            seeds=self.buffers[buffer_idx]["seeds"],
            code_permalink="https://github.com/maxtaylordavies/feedback-DT/blob/master/src/_datasets.py",
            author="Sabrina McCallum",
            author_email="s2431177@ed.ac.uk",
            missions=self.buffers[buffer_idx]["missions"],
            observations=self.buffers[buffer_idx]["observations"],
            actions=self.buffers[buffer_idx]["actions"],
            rewards=self.buffers[buffer_idx]["rewards"],
            feedback=self.buffers[buffer_idx]["feedback"],
            terminations=self.buffers[buffer_idx]["terminations"],
            truncations=self.buffers[buffer_idx]["truncations"],
            episode_terminals=episode_terminals,
        )

        fp = os.path.join(self.fp, str(self.num_shards))
        log(
            f"writing buffer {buffer_idx} to file {fp}.hdf5 ({len(self.buffers[buffer_idx]['observations'])} steps)",
            with_tqdm=True,
        )

        md.save(fp)
        self.num_shards += 1

    def _add_to_buffer(
        self,
        buffer_idx,
        observation,
        action,
        reward,
        feedback,
        next_observation,
        terminated,
        truncated,
        seed,
        mission,
    ):
        """
        Record an environment step in one of the buffers.

        Args:
            buffer_idx (int): The index of the buffer to add the step to.
            observation (np.ndarray): The observation.
            action (int): The action.
            reward (float): The reward.
            feedback (str): The feedback.
            next_observation (np.ndarray): The next observation.
            terminated (bool): Whether the episode terminated.
            truncated (bool): Whether the episode was truncated.
            seed (int): The seed.
            mission (str): The mission.
        """
        self.buffers[buffer_idx]["seeds"][self.steps[buffer_idx]] = seed
        self.buffers[buffer_idx]["missions"][self.steps[buffer_idx]] = mission
        self.buffers[buffer_idx]["observations"][self.steps[buffer_idx]] = observation
        self.buffers[buffer_idx]["actions"][self.steps[buffer_idx]] = action

        self.steps[buffer_idx] += 1
        self.total_steps += 1

        self.buffers[buffer_idx]["rewards"][self.steps[buffer_idx]] = reward
        self.buffers[buffer_idx]["feedback"][self.steps[buffer_idx]] = feedback
        self.buffers[buffer_idx]["terminations"][self.steps[buffer_idx]] = terminated
        self.buffers[buffer_idx]["truncations"][self.steps[buffer_idx]] = truncated

        if terminated or truncated:
            self.buffers[buffer_idx]["seeds"][self.steps[buffer_idx]] = seed
            self.buffers[buffer_idx]["missions"][self.steps[buffer_idx]] = mission
            self.buffers[buffer_idx]["observations"][
                self.steps[buffer_idx]
            ] = next_observation
            self.ep_counts[buffer_idx] += 1
            self.total_episodes += 1

    def _create_episode(self, seed, buffer_idx=0):
        """
        Create an episode in the environment and record it.

        Args:
            seed (int): The seed.
            buffer_idx (int, optional): The index of the buffer to add the episode to. Defaults to 0.
        """
        partial_obs, _ = self.env.reset(seed=seed)
        self.env = FeedbackEnv(
            env=self.env,
            feedback_mode=self.args["feedback_mode"],
            max_steps=self.max_steps,
        )
        terminated, truncated = False, False
        while not (terminated or truncated):
            obs = get_minigrid_obs(
                self.env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
            )
            mission = partial_obs["mission"]
            action = np.random.randint(0, 6)  # random policy

            # execute action
            partial_obs, reward, terminated, truncated, feedback = self.env.step(action)
            reward = (
                float(feedback) if self.args["feedback_mode"] == "numerical" else reward
            )
            next_obs = get_minigrid_obs(
                self.env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
            )

            self._add_to_buffer(
                buffer_idx=buffer_idx,
                observation=obs["image"],
                action=action,
                reward=reward,
                feedback=feedback,
                next_observation=next_obs["image"],
                terminated=terminated,
                truncated=truncated,
                seed=seed,
                mission=mission,
            )

    def _initialise_new_dataset(self):
        """Create a new folder to store the dataset."""
        try:
            if os.path.exists(self.fp):
                print("Overwriting existing dataset folder")
                shutil.rmtree(self.fp, ignore_errors=True)
            os.makedirs(self.fp)
        except:
            if os.path.exists(self.fp):
                print("Overwriting existing dataset folder")
                shutil.rmtree(self.fp, ignore_errors=True)
            os.makedirs(self.fp)

    def _generate_new_dataset(self):
        """Generate a new training dataset based on the user-supplied arguments."""
        total_episodes = self.num_train_seeds * self.eps_per_seed
        pbar = tqdm(total=total_episodes, desc="Generating dataset")

        current_episode, done = 0, False
        while not done:
            for seed in self.train_seeds:
                done = current_episode >= total_episodes

                # start looping through train seeds again
                if done:
                    break

                # create and initialise environment
                log("creating env", with_tqdm=True)
                self.env = gym.make(self.train_config)
                partial_obs, _ = self.env.reset(seed=seed)
                obs = get_minigrid_obs(
                    self.env,
                    partial_obs,
                    self.args["fully_obs"],
                    self.args["rgb_obs"],
                )["image"]
                self.state_dim = np.prod(obs.shape)

                # initialise buffers to store replay data
                if current_episode == 0:
                    self._initialise_buffers(
                        num_buffers=1,
                        obs_shape=obs.shape,
                    )

                # create another episode
                self._create_episode(seed)

                current_episode += 1

                # if buffer contains eps_per_shard episodes or this is final episode, save data to file and clear buffer
                if (self.ep_counts[0] % self.eps_per_shard == 0) or (
                    current_episode >= total_episodes
                ):
                    self._flush_buffer(buffer_idx=0, obs_shape=obs.shape)

                pbar.update(1)
                pbar.refresh()

                if seed == self.train_seeds[-1]:
                    break

        if hasattr(self, "env"):
            self.env.close()
        self._flush_buffer(buffer_idx=0, obs_shape=obs.shape)

    def get_shard_list(self):
        """Set shuffled list of shards to load from."""
        shard_list = np.random.choice(
            np.arange(self.num_shards),
            size=self.num_shards,
        )
        self.shard_list = shard_list.tolist()

    def load_shard(self, idx=None):
        """
        Load a shard from file.

        Args:
            idx (int, optional): The index of the shard to load. Defaults to None (and loads a random shard).
        """
        if idx is None:
            try:
                idx = self.shard_list.pop()
            except IndexError:
                print("Resetting shard list")
                self.get_shard_list()
                idx = self.shard_list.pop()
        self.shard = MinariDataset.load(os.path.join(self.fp, str(idx)))

        # compute start and end timesteps for each episode
        self.episode_ends = np.where(
            self.shard.terminations + self.shard.truncations == 1
        )[0]
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1] + 1])
        self.episode_lengths = self.episode_ends - self.episode_starts + 1
        self.num_episodes = len(self.episode_starts)

        # store state and action dimensions
        self.state_dim, self.act_dim = (
            np.prod(self.shard.observations.shape[1:]),
            6,
        )

    def sample_episode_indices(self, num_eps, dist="uniform"):
        """
        Sample a set of episode indices from the current shard.

        Args:
            num_eps (int): The number of episode indices to sample.
            dist (str, optional): The distribution to sample from. Defaults to "uniform".

        Returns:
            np.ndarray: The sampled episode indices.
        """
        if not self.shard:
            raise Exception("No shard loaded")

        # define a distribution over episodes in current shard
        if dist == "length":
            probs = self.episode_lengths
        elif dist == "inverse":
            probs = 1 / self.episode_lengths
        else:
            probs = np.ones(self.num_episodes)
        probs = probs / np.sum(probs)

        # then use this distribution to sample episode indices
        return np.random.choice(
            np.arange(self.num_episodes),
            size=min(self.num_episodes, num_eps),
            p=probs,
        )

    def sample_episode(
        self,
        ep_idx,
        gamma,
        length=None,
        random_start=False,
        feedback=True,
        mission=True,
    ):
        """
        Sample a subsequence of an episode from the current shard.

        Args:
            ep_idx (int): The index of the episode to sample from.
            gamma (float): The discount factor.
            length (int, optional): The length of the subsequence to sample. Defaults to None (and samples the whole episode).
            random_start (bool, optional): Whether to sample a random start timestep for the subsequence. Defaults to False.
            feedback (bool, optional): Whether to include feedback in the output. Defaults to True.
            mission (bool, optional): Whether to include the mission in the output. Defaults to True.

        Returns:
            dict: The sampled episode.
        """
        if not self.shard:
            raise Exception("No shard loaded")

        # optionally sample a random start timestep for this episode
        if random_start and length and length != self.max_steps:
            start = self.episode_starts[ep_idx] + (
                np.random.randint(0, self.episode_lengths[ep_idx] - 1)
                if self.episode_lengths[ep_idx] > 1
                else 0
            )

            tmp = start + length - 1
            end = min(tmp, self.episode_ends[ep_idx])
        else:
            end = self.episode_ends[ep_idx]
            tpm = end - length + 1 if length else self.episode_starts[ep_idx]
            start = max(tpm, self.episode_starts[ep_idx])

        assert start <= end, f"start: {start}, end: {end}"

        if length:
            assert (
                end <= start + length - 1
            ), f"end: {end}, start: {start}, length: {length}"
        # for slicing purposes, so that we include the end step too
        end += 1

        s = self.shard.observations[start:end]
        s = normalise(s).reshape(1, -1, self.state_dim)

        a = self.shard.actions[start:end]
        a = to_one_hot(a, self.act_dim).reshape(1, -1, self.act_dim)

        rtg = discounted_cumsum(
            self.shard.rewards[start : self.episode_ends[ep_idx] + 1], gamma=gamma
        )
        rtg = rtg[: end - start].reshape(1, -1, 1)

        f = (
            np.hstack(self.shard.feedback[start:end])
            if feedback
            else np.array(["No feedback available."] * s.shape[1])
        ).reshape(1, -1, 1)

        m = (
            np.hstack(self.shard.missions[start:end])
            if mission
            else np.array(["No mission available."] * s.shape[1])
        ).reshape(1, -1, 1)

        return {
            "timesteps": np.arange(0, end - start).reshape(1, -1),
            "mission": m,
            "states": s,
            "actions": a,
            "rewards": self.shard.rewards[start:end].reshape(1, -1, 1),
            "returns_to_go": rtg,
            "feedback": f,
            "attention_mask": np.ones((1, end - start)),
        }

    def _from_ppo_training(self):
        """Generate a new training dataset by training a PPO agent on the environment and recording its experience."""

        def setup(env, num_seeds):
            self.env = env
            partial_obs, _ = self.env.reset(seed=0)
            obs = get_minigrid_obs(
                self.env,
                partial_obs,
                self.args["fully_obs"],
                self.args["rgb_obs"],
            )["image"]
            self._initialise_buffers(num_buffers=num_seeds, obs_shape=obs.shape)
            self._ppo_best_return = -np.inf
            self._ppo_early_stop_count = 0

        # define callback func for storing data
        def callback(exps, logs, seeds):
            obss = exps.obs.image.cpu().numpy()
            actions = exps.action.cpu().numpy().reshape(-1, 1)
            rewards = exps.reward.cpu().numpy().reshape(-1, 1)
            feedback = exps.feedback.reshape(-1, 1)  # feedback is already a numpy array
            next_obss = exps.next_obs.image.cpu().numpy()
            terminations = exps.terminations.cpu().numpy().reshape(-1, 1)
            truncations = exps.truncations.cpu().numpy().reshape(-1, 1)

            # compute average episode return
            ep_end_indices = np.where(terminations + truncations > 0)[0]
            avg_ep_return = np.sum(rewards[ep_end_indices]) / len(ep_end_indices)

            # reshape tensors to be (num_seeds, num_timesteps_per_seed, ...)
            tensors = [
                obss,
                actions,
                rewards,
                feedback,
                next_obss,
                terminations,
                truncations,
            ]
            for i, tensor in enumerate(tensors):
                tensors[i] = tensor.reshape(len(seeds), -1, *tensor.shape[1:])
            (
                obss,
                actions,
                rewards,
                feedback,
                next_obss,
                terminations,
                truncations,
            ) = tensors

            for i, seed in enumerate(seeds):
                for t in range(obss.shape[1]):
                    # process partial observations
                    obs = get_minigrid_obs(
                        self.env,
                        obss[i, t],
                        self.args["fully_obs"],
                        self.args["rgb_obs"],
                    )["image"]
                    next_obs = get_minigrid_obs(
                        self.env,
                        next_obss[i, t],
                        self.args["fully_obs"],
                        self.args["rgb_obs"],
                    )["image"]

                    # determine reward
                    r = (
                        float(feedback[i, t])
                        if self.args["feedback_mode"] == "numerical"
                        else rewards[i, t]
                    )

                    # add step to buffer i
                    self._add_to_buffer(
                        buffer_idx=i,
                        action=actions[i, t],
                        observation=obs,
                        reward=r,
                        feedback=feedback[i, t],
                        next_observation=next_obs,
                        terminated=terminations[i, t],
                        truncated=truncations[i, t],
                        seed=seed,
                        mission=self.env.get_mission(),
                    )

                    # if buffer i is full, flush it
                    if (
                        terminations[i, t]
                        or truncations[i, t]
                        and self.ep_counts[i] >= self.eps_per_shard
                    ):
                        self._flush_buffer(buffer_idx=i, obs_shape=obs.shape)

            log(
                f"total_steps:{self.total_steps}  |  eps:{self.ep_counts}  |  steps:{self.steps}",
                with_tqdm=True,
            )

            if (
                avg_ep_return
                > self._ppo_best_return + self.args["ppo_early_stopping_threshold"]
            ):
                log(f"new best avg ppo return: {avg_ep_return}", with_tqdm=True)
                self._ppo_best_return = avg_ep_return
                self._ppo_early_stop_count = 0
            else:
                self._ppo_early_stop_count += 1
                log(
                    f"incremented early stop count to {self._ppo_early_stop_count}",
                    with_tqdm=True,
                )

            if self._ppo_early_stop_count >= self.args["ppo_early_stopping_patience"]:
                log("early stopping PPO training", with_tqdm=True)
                return True

            # return True if we've collected enough episodes for this config
            return self.total_steps >= self.args["num_steps"]

        log(f"using seeds: {self.train_seeds}", with_tqdm=True)

        # train PPO agent
        ppo = PPOAgent(
            env_name=self.train_config,
            seeds=self.train_seeds,
            feedback_mode=self.args["feedback_mode"],
            max_steps=self.max_steps,
        )
        setup(ppo.env, len(self.train_seeds))
        ppo._train_agent(callback=callback)

        # flush any remaining data to file
        for i in range(len(self.buffers)):
            if self.ep_counts[i] > 0:
                self._save_buffer_to_minari_file(i)

        return self

    def __len__(self):
        """Get the number of episodes in the dataset."""
        return (
            self.num_train_seeds * self.eps_per_seed
            if self.args["policy"] == "random"
            else self.num_shards * self.eps_per_shard
        )

    # ----- these methods aren't used, but need to be defined for torch dataloaders to work -----

    def __getitem__(self, idx):
        return idx

    def __getitems__(self, idxs):
        return idxs

    # -------------------------------------------------------------------------------------------

    @classmethod
    def get_dataset(cls, args):
        return cls(args)._get_dataset()
