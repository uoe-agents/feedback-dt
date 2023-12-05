import os
import pathlib
import warnings
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import h5py
import numpy as np


class Transition:
    def __init__(
        self,
        seeds: np.ndarray,
        observation_shape: Sequence[int],
        action_size: int,
        mission: np.ndarray,
        observation: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        feedback: np.ndarray,
        next_observation: np.ndarray,
        termination: float,
        truncation: float,
        prev_transition: Optional["Transition"] = ...,
        next_transition: Optional["Transition"] = ...,
    ):
        ...

    def get_observation_shape(self) -> Sequence[int]:
        ...

    def get_action_size(self) -> int:
        ...

    @property
    def is_discrete(self) -> bool:
        ...

    @property
    def seeds(self) -> np.ndarray:
        ...

    @property
    def mission(self) -> bool:
        ...

    @property
    def observation(self) -> np.ndarray:
        ...

    @property
    def action(self) -> Union[int, np.ndarray]:
        ...

    @property
    def reward(self) -> float:
        ...

    @property
    def feedback(self) -> np.ndarray:
        ...

    @property
    def next_observation(self) -> np.ndarray:
        ...

    @property
    def termination(self) -> float:
        ...

    @property
    def truncation(self) -> float:
        ...

    @property
    def prev_transition(self) -> Optional["Transition"]:
        ...

    @prev_transition.setter
    def prev_transition(self, transition: "Transition") -> None:
        ...

    @property
    def next_transition(self) -> Optional["Transition"]:
        ...

    @next_transition.setter
    def next_transition(self, transition: "Transition") -> None:
        ...

    def clear_links(self) -> None:
        ...


class TransitionMiniBatch:
    def __init__(
        self,
        transitions: List[Transition],
        n_frames: int = ...,
        n_steps: int = ...,
        gamma: float = ...,
    ):
        ...

    @property
    def missions(self) -> np.ndarray:
        ...

    @property
    def observations(self) -> np.ndarray:
        ...

    @property
    def actions(self) -> np.ndarray:
        ...

    @property
    def rewards(self) -> np.ndarray:
        ...

    @property
    def feedback(self) -> np.ndarray:
        ...

    @property
    def next_observations(self) -> np.ndarray:
        ...

    @property
    def transitions(self) -> List[Transition]:
        ...

    @property
    def terminations(self) -> np.ndarray:
        ...

    @property
    def truncations(self) -> np.ndarray:
        ...

    @property
    def n_steps(self) -> np.ndarray:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Transition]:
        ...


def trace_back_and_clear(transition: Transition) -> None:
    ...


# (o_t, a_t, r_t, t_t)
# r_t = r(o_t, a_t)
# observations = [o_t, o_t+1, o_t+2, o_t, o_t+1]
# actions      = [a_t, a_t+1, a_t+2, a_t, a_t+1]
# rewards      = [r_t, r_t+1, r_t+2, r_t, r_t+1]
# terminals    = [  0,     0,     1,   0,     1]

# obs          = [o_t  , o_t+1, o_t  ]
# next_obs     = [o_t+1, o_t+2, o_t+1]
# action       = [a_t  , a_t+1, a_t  ]
# rewards      = [r_t  , r_t+1, r_t  ]
# terminals    = [0    , 1    , 1    ]


def _safe_size(array):
    if isinstance(array, (list, tuple)):
        return len(array)
    elif isinstance(array, np.ndarray):
        return array.shape[0]
    raise ValueError


def _to_episodes(
    seeds,
    observation_shape,
    action_size,
    missions,
    observations,
    actions,
    rewards,
    feedback,
    terminations,
    truncations,
    episode_terminals,
):
    rets = []
    head_index = 0
    for i in range(_safe_size(observations)):
        if episode_terminals[i]:
            episode = Episode(
                seeds=seeds[head_index : i + 1],
                observation_shape=observation_shape,
                action_size=action_size,
                missions=missions[head_index : i + 1],
                observations=observations[head_index : i + 1],
                actions=actions[head_index : i + 1],
                rewards=rewards[head_index : i + 1],
                feedback=feedback[head_index : i + 1],
                termination=terminations[i],
                truncation=truncations[i],
            )
            rets.append(episode)
            head_index = i + 1
    return rets


def _to_transitions(
    seeds,
    observation_shape,
    action_size,
    observations,
    missions,
    actions,
    rewards,
    feedback,
    termination,
    truncation,
):
    rets = []
    num_data = _safe_size(observations)
    prev_transition = None
    for i in range(num_data):
        seeds = seeds[i]
        mission = missions[i]
        observation = observations[i]
        action = actions[i]
        reward = rewards[i]
        feedback = feedback[i]

        if i == num_data - 1:
            if termination or truncation:
                # dummy observation
                next_observation = np.zeros_like(observation)
            else:
                # skip the last step if not terminated
                break
        else:
            next_observation = observations[i + 1]

        transition = Transition(
            seeds=seeds,
            observation_shape=observation_shape,
            action_size=action_size,
            mission=mission,
            observation=observation,
            action=action,
            reward=reward,
            feedback=feedback,
            next_observation=next_observation,
            termination=termination,
            truncation=truncation,
            prev_transition=prev_transition,
        )

        # set pointer to the next transition
        if prev_transition:
            prev_transition.next_transition = transition

        prev_transition = transition

        rets.append(transition)
    return rets


def _check_discrete_action(actions):
    float_actions = np.array(actions, dtype=np.float32)
    int_actions = np.array(actions, dtype=np.int32)
    return np.all(float_actions == int_actions)


class MinariDataset:
    """Markov-Decision Process Dataset class.

    MinariDataset is deisnged for reinforcement learning datasets to use them like
    supervised learning datasets.

    Adapted from https://github.com/Farama-Foundation/Minari

    .. code-block:: python

        from minari.dataset import MinariDataset

        # 1000 steps of observations with shape of (100,)
        observations = np.random.random((1000, 100))
        # 1000 steps of actions with shape of (4,)
        actions = np.random.random((1000, 4))
        # 1000 steps of rewards
        rewards = np.random.random(1000)
        # 1000 steps of terminal flags
        terminals = np.random.randint(2, size=1000)

        dataset = MinariDataset(observations, actions, rewards, terminals)

    The MinariDataset object automatically splits the given data into list of
    :class:`minari.dataset.Episode` objects.
    Furthermore, the MinariDataset object behaves like a list in order to use with
    scikit-learn utilities.

    .. code-block:: python

        # returns the number of episodes
        len(dataset)

        # access to the first episode
        episode = dataset[0]

        # iterate through all episodes
        for episode in dataset:
            pass

    Args:
        observations (numpy.ndarray): N-D array. If the
            observation is a vector, the shape should be
            `(N, dim_observation)`. If the observations is an image, the shape
            should be `(N, C, H, W)`.
        actions (numpy.ndarray): N-D array. If the actions-space is
            continuous, the shape should be `(N, dim_action)`. If the
            action-space is discrete, the shape should be `(N,)`.
        rewards (numpy.ndarray): array of scalar rewards. The reward function
            should be defined as :math:`r_t = r(s_t, a_t)`.
        terminals (numpy.ndarray): array of binary terminal flags.
        episode_terminals (numpy.ndarray): array of binary episode terminal
            flags. The given data will be split based on this flag.
            This is useful if you want to specify the non-environment
            terminations (e.g. timeout). If ``None``, the episode terminations
            match the environment terminations.
        discrete_action (bool): flag to use the given actions as discrete
            action-space actions. If ``None``, the action type is automatically
            determined.

    """

    def __init__(
        self,
        level_group,
        level_name,
        train_config,
        dataset_name,
        policy,
        feedback_mode,
        seeds,
        code_permalink,
        author,
        author_email,
        missions,
        observations,
        actions,
        rewards,
        feedback,
        terminations,
        truncations,
        episode_terminals=None,
        discrete_action=None,
    ):
        self._level_group = level_group
        self._level_name = level_name
        self._train_config = train_config
        self._dataset_name = dataset_name
        self._policy = policy
        self._feedback_mode = feedback_mode
        self._code_permalink = code_permalink
        self._author = author
        self._author_email = author_email

        # NoneType warnings
        if code_permalink is None:
            warnings.warn(
                "`code_permalink` is set to None. For reproducibility purposes it is highly recommended to link your dataset to versioned code.",
                UserWarning,
            )
        if author is None:
            warnings.warn(
                "`author` is set to None. For longevity purposes it is highly recommended to provide an author name.",
                UserWarning,
            )
        if author_email is None:
            warnings.warn(
                "`author_email` is set to None. For longevity purposes it is highly recommended to provide an author email, or some other obvious contact information.",
                UserWarning,
            )

        # validation
        assert isinstance(observations, np.ndarray), "Observations must be numpy array."
        if len(observations.shape) == 4:
            assert (
                observations.dtype == np.uint8
            ), "Image observation must be uint8 array."
        else:
            if observations.dtype != np.float32:
                observations = np.asarray(observations, dtype=np.float32)

        # check nan
        assert np.all(np.logical_not(np.isnan(observations)))
        assert np.all(np.logical_not(np.isnan(actions)))
        assert np.all(np.logical_not(np.isnan(rewards)))
        assert np.all(np.logical_not(np.isnan(terminations)))
        assert np.all(np.logical_not(np.isnan(truncations)))

        self._seeds = seeds
        self._missions = missions
        self._observations = observations
        self._rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self._feedback = feedback
        self._terminations = np.asarray(terminations, dtype=np.float32).reshape(-1)
        self._truncations = np.asarray(truncations, dtype=np.float32).reshape(-1)

        if episode_terminals is None:
            # if None, episode terminals match the environment terminals
            self._episode_terminals = self._terminations
        else:
            self._episode_terminals = np.asarray(
                episode_terminals, dtype=np.float32
            ).reshape(-1)

        # automatic action type detection
        if discrete_action is None:
            discrete_action = _check_discrete_action(actions)

        self.discrete_action = discrete_action
        if discrete_action:
            self._actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        else:
            self._actions = np.asarray(actions, dtype=np.float32)

        self._episodes = None

    @property
    def level_group(self):
        return self._level_group

    @property
    def level_name(self):
        return self._level_name

    @property
    def train_config(self):
        return self._train_config

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def policy(self):
        return self._policy

    @property
    def feedback_mode(self):
        return self._feedback_mode

    @property
    def seeds(self):
        return self._seeds

    @property
    def code_permalink(self):
        return self._code_permalink

    @property
    def author(self):
        return self._author

    @property
    def author_email(self):
        return self._author_email

    @property
    def missions(self):
        return self._missions

    @property
    def observations(self):
        """Returns the observations.

        Returns:
            numpy.ndarray: array of observations.

        """
        return self._observations

    @property
    def actions(self):
        """Returns the actions.

        Returns:
            numpy.ndarray: array of actions.

        """
        return self._actions

    @property
    def rewards(self):
        """Returns the rewards.

        Returns:
            numpy.ndarray: array of rewards

        """
        return self._rewards

    @property
    def feedback(self):
        return self._feedback

    @property
    def terminations(self):
        """Returns the terminations flags.

        Returns:
            numpy.ndarray: array of terminal flags.

        """
        return self._terminations

    @property
    def truncations(self):
        """Returns the terminations flags.

        Returns:
            numpy.ndarray: array of terminal flags.

        """
        return self._truncations

    @property
    def episode_terminals(self):
        """Returns the episode terminal flags.

        Returns:
            numpy.ndarray: array of episode terminal flags.

        """
        return self._episode_terminals

    @property
    def episodes(self):
        """Returns the episodes.

        Returns:
            list(minari.dataset.Episode):
                list of :class:`minari.dataset.Episode` objects.

        """
        if self._episodes is None:
            self.build_episodes()
        return self._episodes

    def size(self):
        """Returns the number of episodes in the dataset.

        Returns:
            int: the number of episodes.

        """
        return len(self.episodes)

    def get_action_size(self):
        """Returns dimension of action-space.

        If `discrete_action=True`, the return value will be the maximum index
        +1 in the give actions.

        Returns:
            int: dimension of action-space.

        """
        if self.discrete_action:
            return int(np.max(self._actions) + 1)
        return self._actions.shape[1]

    def get_observation_shape(self):
        """Returns observation shape.

        Returns:
            tuple: observation shape.

        """
        return self._observations[0].shape

    def is_action_discrete(self):
        """Returns `discrete_action` flag.

        Returns:
            bool: `discrete_action` flag.

        """
        return self.discrete_action

    def compute_stats(self):
        """Computes statistics of the dataset.

        .. code-block:: python

            stats = dataset.compute_stats()

            # return statistics
            stats['return']['mean']
            stats['return']['std']
            stats['return']['min']
            stats['return']['max']

            # reward statistics
            stats['reward']['mean']
            stats['reward']['std']
            stats['reward']['min']
            stats['reward']['max']

            # action (only with continuous control actions)
            stats['action']['mean']
            stats['action']['std']
            stats['action']['min']
            stats['action']['max']

            # observation (only with numpy.ndarray observations)
            stats['observation']['mean']
            stats['observation']['std']
            stats['observation']['min']
            stats['observation']['max']

        Returns:
            dict: statistics of the dataset.

        """
        episode_returns = []
        for episode in self.episodes:
            episode_returns.append(episode.compute_return())

        stats = {
            "return": {
                "mean": np.mean(episode_returns),
                "std": np.std(episode_returns),
                "min": np.min(episode_returns),
                "max": np.max(episode_returns),
                "histogram": np.histogram(episode_returns, bins=20),
            },
            "reward": {
                "mean": np.mean(self._rewards),
                "std": np.std(self._rewards),
                "min": np.min(self._rewards),
                "max": np.max(self._rewards),
                "histogram": np.histogram(self._rewards, bins=20),
            },
        }

        # only for continuous control task
        if not self.discrete_action:
            # calculate histogram on each dimension
            hists = []
            for i in range(self.get_action_size()):
                hists.append(np.histogram(self.actions[:, i], bins=20))
            stats["action"] = {
                "mean": np.mean(self.actions, axis=0),
                "std": np.std(self.actions, axis=0),
                "min": np.min(self.actions, axis=0),
                "max": np.max(self.actions, axis=0),
                "histogram": hists,
            }
        else:
            # count frequency of discrete actions
            freqs = []
            for i in range(self.get_action_size()):
                freqs.append((self.actions == i).sum())
            stats["action"] = {"histogram": [freqs, np.arange(self.get_action_size())]}

        # avoid large copy when observations are huge data.
        stats["observation"] = {
            "mean": np.mean(self.observations, axis=0),
            "std": np.std(self.observations, axis=0),
            "min": np.min(self.observations, axis=0),
            "max": np.max(self.observations, axis=0),
        }

        return stats

    def append(
        self,
        seeds,
        missions,
        observations,
        actions,
        rewards,
        feedback,
        terminations,
        truncations,
        episode_terminals=None,
    ):
        """Appends new data.

        Args:
            missions (numpy.ndarray): missions.
            observations (numpy.ndarray): N-D array.
            actions (numpy.ndarray): actions.
            rewards (numpy.ndarray): rewards.
            feedback (numpy.ndarray): feedback.
            terminals (numpy.ndarray): terminals.
            episode_terminals (numpy.ndarray): episode terminals.

        """
        # validation
        for observation, action in zip(observations, actions):
            assert (
                observation.shape == self.get_observation_shape()
            ), f"Observation shape must be {self.get_observation_shape()}."
            if self.discrete_action:
                if int(action) >= self.get_action_size():
                    message = (
                        f"New action size is higher than" f" {self.get_action_size()}."
                    )
                    warnings.warn(message)
            else:
                assert action.shape == (
                    self.get_action_size(),
                ), f"Action size must be {self.get_action_size()}."

        self._seeds = np.hstack([self._seeds, seeds])

        self._missions = np.hstack([self._missions, missions])

        self._observations = np.vstack([self._observations, observations])

        if self.discrete_action:
            self._actions = np.hstack([self._actions, actions])
        else:
            self._actions = np.vstack([self._actions, actions])

        self._rewards = np.hstack([self._rewards, rewards])
        self._feedback = np.hstack([self._feedback, feedback])
        self._terminations = np.hstack([self._terminations, terminations])
        if episode_terminals is None:
            episode_terminals = terminations or truncations
        self._episode_terminals = np.hstack(
            [self._episode_terminals, episode_terminals]
        )

        episodes = _to_episodes(
            seeds=self._seeds,
            observation_shape=self.get_observation_shape(),
            action_size=self.get_action_size(),
            missions=self._missions,
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            feedback=self._feedback,
            terminations=self._terminations,
            truncations=self._truncations,
            episode_terminals=self._episode_terminals,
        )

        self._episodes = episodes

    def extend(self, dataset):
        """Extend dataset by another dataset.

        Args:
            dataset (minari.dataset.MinariDataset): dataset.

        """
        assert (
            self.is_action_discrete() == dataset.is_action_discrete()
        ), "Dataset must have discrete action-space."
        assert (
            self.get_observation_shape() == dataset.get_observation_shape()
        ), f"Observation shape must be {self.get_observation_shape()}"

        self.append(
            dataset.seeds,
            dataset.missions,
            dataset.observations,
            dataset.actions,
            dataset.rewards,
            dataset.feedback,
            dataset.terminations,
            dataset.truncations,
            dataset.episode_terminals,
        )

    def save(self, fname=None):
        """Saves dataset as HDF5.

        Args:
            fname (str): file path.

        """
        fp = os.environ.get("MINARI_DATASETS_PATH")
        if not fp:
            fp = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
        os.makedirs(fp, exist_ok=True)
        fp = os.path.join(fp, f"{fname if fname else self._dataset_name}.hdf5")

        with h5py.File(fp, "w") as f:
            f.create_dataset("level_group", data=self._level_group)
            f.create_dataset("level_name", data=self._level_name)
            f.create_dataset("train_config", data=self._train_config)
            f.create_dataset("dataset_name", data=self._dataset_name)
            f.create_dataset("policy", data=self._policy)
            f.create_dataset("feedback_mode", data=self._feedback_mode)
            f.create_dataset(
                "seeds", data=np.asarray(self._seeds, dtype="S"), compression="gzip"
            )
            f.create_dataset(
                "code_permalink", data=str(self._code_permalink)
            )  # allows saving of NoneType
            f.create_dataset("author", data=str(self._author))
            f.create_dataset("author_email", data=str(self._author_email))
            f.create_dataset(
                "missions",
                data=np.asarray(self._missions, dtype="S"),
                compression="gzip",
            )
            f.create_dataset(
                "observations", data=self._observations, compression="gzip"
            )
            f.create_dataset("actions", data=self._actions, compression="gzip")
            f.create_dataset("rewards", data=self._rewards, compression="gzip")
            f.create_dataset(
                "feedback",
                data=np.asarray(self._feedback, dtype="S"),
                compression="gzip",
            )
            f.create_dataset(
                "terminations", data=self._terminations, compression="gzip"
            )
            f.create_dataset("truncations", data=self._truncations, compression="gzip")
            f.create_dataset(
                "episode_terminals", data=self._episode_terminals, compression="gzip"
            )
            f.create_dataset("discrete_action", data=self.discrete_action)
            f.create_dataset("version", data="1.0")
            f.flush()

    @classmethod
    def load(cls, fname):
        """Loads dataset from HDF5.

        .. code-block:: python

            import numpy as np
            from minari.dataset import MinariDataset

            dataset = MinariDataset(np.random.random(10, 4),
                                 np.random.random(10, 2),
                                 np.random.random(10),
                                 np.random.randint(2, size=10))

            # save as HDF5
            dataset.dump('dataset.h5')

            # load from HDF5
            new_dataset = MinariDataset.load('dataset.h5')

        Args:
            fname (str): file path.

        """
        fp = os.environ.get("MINARI_DATASETS_PATH")
        if not fp:
            fp = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
        fp = os.path.join(fp, f"{fname}.hdf5")

        with h5py.File(fp, "r") as f:
            level_group = str(f["level_group"][()], "utf-8")
            level_name = str(f["level_name"][()], "utf-8")
            train_config = str(f["train_config"][()], "utf-8")
            dataset_name = str(f["dataset_name"][()], "utf-8")
            policy = str(f["policy"][()], "utf-8")
            feedback_mode = str(f["feedback_mode"][()], "utf-8")
            seeds = np.char.decode(f["seeds"][()])
            code_permalink = str(f["code_permalink"][()], "utf-8")
            author = str(f["author"][()], "utf-8")
            author_email = str(f["author_email"][()], "utf-8")
            missions = np.char.decode(f["missions"][()])
            observations = f["observations"][()]
            actions = f["actions"][()]
            rewards = f["rewards"][()]
            feedback = np.char.decode(f["feedback"][()])
            terminations = f["terminations"][()]
            truncations = f["truncations"][()]
            discrete_action = f["discrete_action"][()]

            # for backward compatibility
            if "episode_terminals" in f:
                episode_terminals = f["episode_terminals"][()]
            else:
                episode_terminals = None

        return cls(
            level_group=level_group,
            level_name=level_name,
            train_config=train_config,
            dataset_name=dataset_name,
            policy=policy,
            feedback_mode=feedback_mode,
            seeds=seeds,
            code_permalink=code_permalink,
            author=author,
            author_email=author_email,
            missions=missions,
            observations=observations,
            actions=actions,
            rewards=rewards,
            feedback=feedback,
            terminations=terminations,
            truncations=truncations,
            episode_terminals=episode_terminals,
            discrete_action=discrete_action,
        )

    def build_episodes(self):
        """Builds episode objects.

        This method will be internally called when accessing the episodes
        property at the first time.

        """
        self._episodes = _to_episodes(
            seeds=self._seeds,
            observation_shape=self.get_observation_shape(),
            action_size=self.get_action_size(),
            missions=self._missions,
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            feedback=self._feedback,
            terminations=self._terminations,
            truncations=self.truncations,
            episode_terminals=self._episode_terminals,
        )

    @classmethod
    def random(cls, num_eps, ep_length, state_dim, act_dim):
        states = np.random.rand(num_eps * ep_length, state_dim)
        actions = np.random.randint(0, act_dim, size=(num_eps * ep_length))
        rewards = np.random.rand(num_eps * ep_length)

        terminations = np.zeros((num_eps, ep_length))
        terminations[:, -1] = 1
        terminations = terminations.reshape((num_eps * ep_length))
        truncations = np.zeros_like(terminations)

        return cls(
            level_group="",
            level_name="",
            train_config="",
            dataset_name="",
            policy="",
            feedback_mode="",
            seeds=np.array([]),
            code_permalink="",
            author="",
            author_email="",
            missions=np.array([]),
            observations=states,
            actions=actions,
            rewards=rewards,
            feedback=np.array([]),
            terminations=terminations,
            truncations=truncations,
            episode_terminals=None,
            discrete_action=True,
        )

    @classmethod
    def from_dqn_replay(cls, data_dir, game, num_samples):
        obs, acts, rewards, dones = [], [], [], []

        buffer_idx, depleted = -1, True
        while len(obs) < num_samples:
            if depleted:
                buffer_idx, depleted = buffer_idx + 1, False
                buffer, i = load_dopamine_buffer(data_dir, game, 50 - buffer_idx), 0

            (
                s,
                a,
                r,
                _,
                _,
                _,
                terminal,
                _,
            ) = buffer.sample_transition_batch(batch_size=1, indices=[i])

            obs.append(s[0])
            acts.append(a[0])
            rewards.append(r[0])
            dones.append(terminal[0])

            i += 1
            depleted = i == buffer._replay_capacity

        return cls(
            level_group="",
            level_name="",
            train_config="",
            dataset_name=f"dqn_replay-{game}-{num_samples}",
            policy="",
            feedback_mode="",
            seeds=np.array([]),
            code_permalink="",
            author="",
            author_email="",
            missions=np.array([]),
            observations=np.array(obs),
            actions=np.array(acts),
            rewards=np.array(rewards),
            feedback=np.array([]),
            terminations=np.array(dones),
            truncations=np.zeros_like(dones),
            episode_terminals=None,
            discrete_action=True,
        )

    # helper func to load a dopamine buffer from dqn replay logs
    def load_dopamine_buffer(data_dir, game, buffer_idx):
        replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000,
        )
        replay_buffer.load(os.path.join(data_dir, game, "1", "replay_logs"), buffer_idx)
        return replay_buffer

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.episodes[index]

    def __iter__(self):
        return iter(self.episodes)


class Episode:
    """Episode class.

    This class is designed to hold data collected in a single episode.

    Episode object automatically splits data into list of
    :class:`minari.dataset.Transition` objects.
    Also Episode object behaves like a list object for ease of access to
    transitions.

    .. code-block:: python

        # return the number of transitions
        len(episode)

        # access to the first transition
        transitions = episode[0]

        # iterate through all transitions
        for transition in episode:
            pass

    Args:
        observation_shape (tuple): observation shape.
        action_size (int): dimension of action-space.
        observations (numpy.ndarray): observations.
        actions (numpy.ndarray): actions.
        rewards (numpy.ndarray): scalar rewards.
        terminal (bool): binary terminal flag. If False, the episode is not
            terminated by the environment (e.g. timeout).

    """

    def __init__(
        self,
        seeds,
        observation_shape,
        action_size,
        missions,
        observations,
        actions,
        rewards,
        feedback,
        termination=True,
        truncation=True,
    ):
        # validation
        assert isinstance(observations, np.ndarray), "Observation must be numpy array."
        if len(observation_shape) == 3:
            assert (
                observations.dtype == np.uint8
            ), "Image observation must be uint8 array."
        else:
            if observations.dtype != np.float32:
                observations = np.asarray(observations, dtype=np.float32)

        # fix action dtype and shape
        if len(actions.shape) == 1:
            actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        else:
            actions = np.asarray(actions, dtype=np.float32)

        self._seeds = seeds
        self.observation_shape = observation_shape
        self.action_size = action_size
        self._missions = missions
        self._observations = observations
        self._actions = actions
        self._rewards = np.asarray(rewards, dtype=np.float32)
        self._feedback = feedback
        self._termination = termination
        self._truncation = truncation
        self._transitions = None

    @property
    def seeds(self):
        """Returns the seeds.

        Returns:
            numpy.ndarray: array of seeds.
        """
        return self._seeds

    @property
    def missions(self):
        """Returns the missions.

        Returns:
            numpy.ndarray: array of missions.

        """
        return self._missions

    @property
    def observations(self):
        """Returns the observations.

        Returns:
            numpy.ndarray: array of observations.

        """
        return self._observations

    @property
    def actions(self):
        """Returns the actions.

        Returns:
            numpy.ndarray: array of actions.

        """
        return self._actions

    @property
    def rewards(self):
        """Returns the rewards.

        Returns:
            numpy.ndarray: array of rewards.

        """
        return self._rewards

    @property
    def feedback(self):
        """Returns the feedback.

        Returns:
            numpy.ndarray: array of feedback.

        """
        return self._feedback

    @property
    def termination(self):
        """Returns the termination flag.

        Returns:
            bool: the terminal flag.

        """
        return self._termination

    @property
    def truncation(self):
        """Returns the truncation flag.

        Returns:
            bool: the terminal flag.

        """
        return self._truncation

    @property
    def transitions(self):
        """Returns the transitions.

        Returns:
            list(minari.dataset.Transition):
                list of :class:`minari.dataset.Transition` objects.

        """
        if self._transitions is None:
            self.build_transitions()
        return self._transitions

    def build_transitions(self):
        """Builds transition objects.

        This method will be internally called when accessing the transitions
        property at the first time.

        """
        self._transitions = _to_transitions(
            seeds=self.seeds,
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            missions=self._missions,
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            feedback=self._feedback,
            termination=self._termination,
            truncation=self._truncation,
        )

    def size(self):
        """Returns the number of transitions.

        Returns:
            int: the number of transitions.

        """
        return len(self.transitions)

    def get_observation_shape(self):
        """Returns observation shape.

        Returns:
            tuple: observation shape.

        """
        return self.observation_shape

    def get_action_size(self):
        """Returns dimension of action-space.

        Returns:
            int: dimension of action-space.

        """
        return self.action_size

    def compute_return(self):
        """Computes sum of rewards.

        .. math::

            R = \\sum_{i=1} r_i

        Returns:
            float: episode return.

        """
        return np.sum(self._rewards)

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.transitions[index]

    def __iter__(self):
        return iter(self.transitions)
