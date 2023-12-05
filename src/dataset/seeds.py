import json
import os
import random
import threading
from itertools import combinations_with_replacement

import gymnasium as gym
import numpy as np
from jsonc_parser.parser import JsoncParser
from minigrid.core.constants import COLOR_NAMES
from minigrid.envs.babyai.core.verifier import LOC_NAMES
from minigrid.envs.babyai.core.verifier import OBJ_TYPES_NOT_DOOR
from minigrid.envs.babyai.core.verifier import OpenInstr

from src.dataset.custom_feedback_verifier import TaskFeedback

LEVELS_CONFIGS = {
    "original_tasks": {
        "GoToObj": ["BabyAI-GoToObj-v0", "BabyAI-GoToObjS4-v0", "BabyAI-GoToObjS6-v0"],
        "GoToRedBallGrey": ["BabyAI-GoToRedBallGrey-v0"],
        "GoToRedBall": ["BabyAI-GoToRedBall-v0"],
        "GoToLocal": [
            "BabyAI-GoToLocal-v0",
            "BabyAI-GoToLocalS5N2-v0",
            "BabyAI-GoToLocalS6N2-v0",
            "BabyAI-GoToLocalS6N3-v0",
            "BabyAI-GoToLocalS6N4-v0",
            "BabyAI-GoToLocalS7N4-v0",
            "BabyAI-GoToLocalS7N5-v0",
            "BabyAI-GoToLocalS8N2-v0",
            "BabyAI-GoToLocalS8N3-v0",
            "BabyAI-GoToLocalS8N4-v0",
            "BabyAI-GoToLocalS8N5-v0",
            "BabyAI-GoToLocalS8N6-v0",
            "BabyAI-GoToLocalS8N7-v0",
        ],
        "PutNextLocal": [
            "BabyAI-PutNextLocal-v0",
            "BabyAI-PutNextLocalS5N3-v0",
            "BabyAI-PutNextLocalS6N4-v0",
        ],
        "PickupLoc": ["BabyAI-PickupLoc-v0"],
        "GoToObjMaze": [
            "BabyAI-GoToObjMaze-v0",
            "BabyAI-GoToObjMazeS4R2-v0",
            "BabyAI-GoToObjMazeS4-v0",
            "BabyAI-GoToObjMazeS5-v0",
            "BabyAI-GoToObjMazeS6-v0",
            "BabyAI-GoToObjMazeS7-v0",
        ],
        "GoTo": [
            "BabyAI-GoTo-v0",
        ],
        "Pickup": ["BabyAI-Pickup-v0"],
        "UnblockPickup": ["BabyAI-UnblockPickup-v0"],
        "Open": ["BabyAI-Open-v0"],
        "Unlock": ["BabyAI-Unlock-v0"],
        "PutNext": [
            "BabyAI-PutNextS4N1-v0",
            "BabyAI-PutNextS5N1-v0",
            "BabyAI-PutNextS5N2-v0",
            "BabyAI-PutNextS6N3-v0",
            "BabyAI-PutNextS7N4-v0",
        ],
        "Synth": ["BabyAI-Synth-v0"],
        "SynthLoc": ["BabyAI-SynthLoc-v0"],
        "GoToSeq": ["BabyAI-GoToSeq-v0", "BabyAI-GoToSeqS5R2-v0"],
        "SynthSeq": ["BabyAI-SynthSeq-v0"],
        "GoToImpUnlock": ["BabyAI-GoToImpUnlock-v0"],
        "BossLevel": ["BabyAI-BossLevel-v0"],
    },
    "new_tasks": {
        "GoToRedBallNoDists": ["BabyAI-GoToRedBallNoDists-v0"],
        "GoToRedBlueBall": ["BabyAI-GoToRedBlueBall-v0"],
        "GoToDoor": ["BabyAI-GoToDoor-v0"],
        "GoToObjDoor": ["BabyAI-GoToObjDoor-v0"],
        "OpenDoor": [
            "BabyAI-OpenDoor-v0",
            "BabyAI-OpenDoorColor-v0",
            "BabyAI-OpenDoorLoc-v0",
        ],
        "OpenTwoDoors": ["BabyAI-OpenTwoDoors-v0", "BabyAI-OpenRedBlueDoors-v0"],
        "OpenDoorsOrder": ["BabyAI-OpenDoorsOrderN2-v0", "BabyAI-OpenDoorsOrderN4-v0"],
        "PickuDist": ["BabyAI-PickupDist-v0"],
        "PickupAbove": ["BabyAI-PickupAbove-v0"],
        "UnlockLocal": ["BabyAI-UnlockLocal-v0", "BabyAI-UnlockLocalDist-v0"],
        "KeyInBox": ["BabyAI-KeyInBox-v0"],
        "UnlockPickup": ["BabyAI-UnlockPickup-v0", "BabyAI-UnlockPickupDist-v0"],
        "BlockedUnlockPickup": ["BabyAI-BlockedUnlockPickup-v0"],
        "UnlockToUnlock": ["BabyAI-UnlockToUnlock-v0"],
        "ActionObjDoor": ["BabyAI-ActionObjDoor-v0"],
        "FindObj": [
            "BabyAI-FindObjS5-v0",
            "BabyAI-FindObjS6-v0",
            "BabyAI-FindObjS7-v0",
        ],
        "KeyCorridor": [
            "BabyAI-KeyCorridor-v0",
            "BabyAI-KeyCorridorS3R1-v0",
            "BabyAI-KeyCorridorS3R2-v0",
            "BabyAI-KeyCorridorS3R3-v0",
            "BabyAI-KeyCorridorS4R3-v0",
            "BabyAI-KeyCorridorS5R3-v0",
        ],
        "OneRoom": [
            "BabyAI-OneRoomS8-v0",
            "BabyAI-OneRoomS12-v0",
            "BabyAI-OneRoomS16-v0",
            "BabyAI-OneRoomS20-v0",
        ],
        "MoveTwoAcross": ["BabyAI-MoveTwoAcrossS5N2-v0", "BabyAI-MoveTwoAcrossS8N9-v0"],
        "MiniBossLevel": ["BabyAI-MiniBossLevel-v0"],
        "BossLevelNoUnlock": ["BabyAI-BossLevelNoUnlock-v0"],
    },
}


class SeedFinder:
    """
    Class to find in-domain and ood seeds for a given level and config.

    In some cases, this involves all seeds for a given level or config.
    In other cases, this involves a subset of seeds for a given level or config.

    Args:
        n_train_seeds_required (int): number of in-domain train seeds required.
        original_tasks_only (bool): whether to only use the original tasks or also include the new tasks.
    """

    def __init__(self, n_train_seeds_required=10**6, original_tasks_only=True):
        """
        Initialise the SeedFinder class.
        """
        self.n_validation_seeds_required = 128
        self.n_train_seeds_required = n_train_seeds_required
        self.LEVELS_CONFIGS = (
            LEVELS_CONFIGS["original_tasks"]
            if original_tasks_only
            else {**LEVELS_CONFIGS["original_tasks"], **LEVELS_CONFIGS["new_tasks"]}
        )
        random.seed(42)
        self.random_colors = self._pick_random_colors()
        self.random_types = self._pick_random_types()
        self.random_rel_loc = self._pick_random_rel_location()
        self.iid_room_size = 8
        self.iid_num_cols = 3
        self.iid_num_rows = 3
        self.random_room_quadrant = self._pick_random_room_quadrant()
        self.random_maze_quadrant = self._pick_random_maze_quadrant()
        self.ood_types = {
            "size": self._check_size,
            "color_type": self._check_color_type,
            "agent_loc": self._check_agent_loc,
            "rel_loc": self._check_rel_loc,
            "object_task": self._check_object_task,
        }
        self.seeds_dir = os.getenv("SEEDS_DIR") or "seeds"
        if not os.path.exists(self.seeds_dir):
            os.mkdir(self.seeds_dir)

    def _pick_random_colors(self):
        """
        Pick a random color from the list of colors.

        Args:
            i (int): index of the color to pick.

        Returns:
            str: a random object color from a list of colors.
        """
        return random.sample(COLOR_NAMES, 2)

    def _pick_random_types(self):
        """
        Pick a random type from the list of types.

        Args:
            i (int): index of the type to pick.

        Returns:
            str: a random object type from a list of types.
        """
        return random.sample(OBJ_TYPES_NOT_DOOR, 2)

    def _pick_random_rel_location(self):
        """
        Pick a random relative goal location from the list of possible descriptions.

        This applies to Loc levels only.

        Returns:
            str: a random relative goal object location.
        """
        return random.sample(LOC_NAMES, 1)[0]

    def _get_task_list(self, env):
        """
        Get the list of mission tasks for a given level.

        If this is a Sequence type level, then the list of tasks is a list of up to four subtasks.

        If this is not a Sequence type level, then the list of tasks consists of a single task.

        Args:
            env: instance of the level made using a seed.

        Returns:
            list: list of mission tasks.
        """
        return TaskFeedback(env).tasks

    def _is_maze(self, env):
        """
        Check if the level is a maze.

        Args:
            env: instance of the level made using a seed.

        Returns:
            bool: True if the level is a maze, False otherwise.
        """
        return env.unwrapped.num_rows > 1 or env.unwrapped.num_cols > 2

    def _get_all_positions(self, size, n):
        """
        Get all possible positions in the grid for a given axis.

        Args:
            env: instance of the level made using a seed.
            size (int): size of the room.
            n (int): number of rows or columns in the grid.

        Returns:
            list: list of tuples of x or y coordinates of all possible positions in the grid.
        """
        all_positions = [c for c in range(1, size * n - n) if c % (size - 1) != 0]
        return np.asarray(all_positions)

    def _split_positions_by_room(self, positions, n):
        """
        Split an array of positions into arrays of quadrants.

        Args:
            positions (np.array): array of x or y coordinates of all possible positions in the grid.
            n (int): number of rows or columns in the grid.

        Returns:
            array: array of arrays of x or y coordinates of all possible positions in the grid, by room.
        """
        return np.array_split(
            np.asarray(positions),
            n,
        )

    def _split_rooms_into_quadrants(self, room_positions):
        """
        Split an array of room positions into arrays of quadrants.

        Args:
            room_positions (np.array): array of x or y coordinates of all possible positions in the grid, by room.

        Returns:
            array: array of arrays of x or y coordinates of all possible positions in the grid, by room quadrant.
        """
        return [np.array_split(x, 2) for x in room_positions]

    def _get_quadrant_lookup(self, quadrant_positions):
        """
        Get a lookup of quadrants to positions.

        Args:
            quadrant_positions (list): list of arrays of x or y coordinates of all possible positions in the grid, by room quadrant.

        Returns:
            dict: dict of quadrants to positions.
        """
        quadrant_lookup = {}
        for _, quadrants in enumerate(quadrant_positions):
            for quadrant, positions in enumerate(quadrants):
                if quadrant + 1 not in quadrant_lookup:
                    quadrant_lookup[quadrant + 1] = list(positions)
                else:
                    quadrant_lookup[quadrant + 1].extend(list(positions))
        return quadrant_lookup

    def _agent_pos_to_room_quadrant(self, env):
        """
        Convert the agent position into a quadrant in the room.

        Args:
            env: instance of the level made using a seed.

        Returns:
            tuple: tuple of x and y coordinates of the quadrant of the room.

        """
        all_positions_x = self._get_all_positions(
            env.unwrapped.room_size, env.unwrapped.num_cols
        )
        all_positions_y = self._get_all_positions(
            env.unwrapped.room_size, env.unwrapped.num_rows
        )

        room_positions_x = self._split_positions_by_room(
            all_positions_x, env.unwrapped.num_cols
        )
        room_positions_y = self._split_positions_by_room(
            all_positions_y, env.unwrapped.num_rows
        )

        quadrant_positions_x = self._split_rooms_into_quadrants(room_positions_x)
        quadrant_positions_y = self._split_rooms_into_quadrants(room_positions_y)

        for x_quadrant, x_positions in self._get_quadrant_lookup(
            quadrant_positions_x
        ).items():
            for y_quadrant, y_positions in self._get_quadrant_lookup(
                quadrant_positions_y
            ).items():
                if (
                    env.unwrapped.agent_pos[0] in x_positions
                    and env.unwrapped.agent_pos[1] in y_positions
                ):
                    return x_quadrant, y_quadrant

    def _agent_pos_to_maze_quadrant(self, env):
        """
        Convert the agent position into a quadrant in the maze.

        Args:
            env: instance of the level made using a seed.

        Returns:
            tuple: tuple of x and y coordinates of the quadrant of the maze.
        """

        all_positions_x = self._get_all_positions(
            env.unwrapped.room_size, env.unwrapped.num_cols
        )
        all_positions_y = self._get_all_positions(
            env.unwrapped.room_size, env.unwrapped.num_rows
        )

        room_positions_x = self._split_positions_by_room(
            all_positions_x, env.unwrapped.num_cols
        )
        room_positions_y = self._split_positions_by_room(
            all_positions_y, env.unwrapped.num_rows
        )

        for x_room, x_positions in enumerate(room_positions_x):
            for y_room, y_positions in enumerate(room_positions_y):
                if (
                    env.unwrapped.agent_pos[0] in x_positions
                    and env.unwrapped.agent_pos[1] in y_positions
                ):
                    return x_room, y_room

    def _get_agent_quadrants(self, env):
        """
        Get the room and maze quadrants of the agent in the grid.

        Args:
            env: instance of the level made using a seed.

        Returns:
            tuple: tuple of tuple of x and y quadrant of the agent in the room and maze.
        """
        if self._is_maze(env):
            return self._agent_pos_to_room_quadrant(
                env
            ), self._agent_pos_to_maze_quadrant(env)
        return self._agent_pos_to_room_quadrant(env), None

    def _get_possible_room_quadrants(self):
        """
        Get a list of possible room grid quadrants.

        Args:
            env: instance of the level made using a seed.

        Returns:
            list: list of tuples of x and y coordinates for all possible quadrants of the room.
        """
        return [q for q in combinations_with_replacement([1, 2], 2)]

    def _get_possible_maze_quadrants(self):
        """
        Get a list of possible maze grid quadrants.

        Args:
            env: instance of the level made using a seed.

        Returns:
            list: list of tuples of x and y coordinates for all possible quadrants of the maze.
        """
        return [
            q
            for q in combinations_with_replacement(
                [self.iid_num_cols, self.iid_num_rows], 2
            )
        ]

    def _pick_random_room_quadrant(self):
        """
        Pick a random room quadrant from a list of possible room quadrants.

        Returns:
            tuple: tuple of x and y coordinates for a randomly picked quadrant of the room.
        """
        return random.sample(
            self._get_possible_room_quadrants(),
            k=1,
        )[0]

    def _pick_random_maze_quadrant(self):
        """
        Pick a random maze quadrant from a list of possible maze quadrants.

        Returns:
            tuple: tuple of x and y coordinates for a randomly picked quadrant of the maze.
        """
        return random.sample(
            self._get_possible_maze_quadrants(),
            k=1,
        )[0]

    def _goal_doors_locked(self, env, task):
        """
        Check if the goal doors are locked.

        Args:
            env: instance of the level made using a seed.
            task: instance of the subtask for the env that is to be checked.

        Returns:
            bool: True if any of the goal doors are locked, False otherwise.
        """
        for pos in task.desc.obj_poss:
            cell = env.unwrapped.grid.get(*pos)
            if cell and cell.type == "door" and cell.is_locked:
                return True
        return False

    def _has_two_doors_unlock(self, env):
        """
        Check if the task explicitly requires unlocking two doors.

        Args:
            env: instance of the level made using a seed.

        Returns:
            bool: True if the level has two goal doors to unlock, False otherwise.
        """
        if len(self._get_task_list(env)) > 1:
            door_count = 0
            for task in self._get_task_list(env):
                try:
                    if (
                        isinstance(task, OpenInstr)
                        and task.desc.type == "door"
                        and self._goal_doors_locked(env, task)
                    ):
                        door_count += 1
                except AttributeError:
                    continue
            if door_count > 1:
                return True
        return False

    def _check_size(self, env):
        """
        Check if the level contains the unseen size of the room or maze.

        Args:
            env: instance of the level made using a seed.

        Returns:
            bool: True if the room size is 8 and maze size is 3x3, False otherwise.
        """
        if self._is_maze(env):
            return (
                env.unwrapped.room_size != self.iid_room_size
                or env.unwrapped.num_cols != self.iid_num_cols
                or env.unwrapped.num_rows != self.iid_num_rows
            )
        return env.unwrapped.room_size != self.iid_room_size

    def _check_color_type(self, env):
        """
        Check if the level contains the unseen color-type combinations for goal objects.

        Args:
            env: instance of the level made using a seed.

        Returns:
            bool: True if (any of) the goal object/s is/are of the unseen color and type, False otherwise.
        """
        for task in self._get_task_list(env):
            try:
                if (
                    task.desc.color == self.random_colors[0]
                    and task.desc.type == self.random_types[0]
                ):
                    return True
            except AttributeError:
                if (
                    task.desc_move.color == self.random_colors[0]
                    and task.desc_move.type == self.random_types[0]
                ) or (
                    task.desc_fixed.color == self.random_colors[0]
                    and task.desc_fixed.type == self.random_types[0]
                ):
                    return True
        return False

    def _check_agent_loc(self, env):
        """
        Check if the level contains an agent starting position in an unseen room (and for mazes, maze) quadrant(s).

        Args:
            env: instance of the level made using a seed.

        Returns:
            bool: True if agent starting location is in the unseed quadrant in the room (single room) and maze (for maze only), False otherwise.
        """
        room_quadrant, maze_quadrant = self._get_agent_quadrants(env)
        if room_quadrant == self.random_room_quadrant or (
            maze_quadrant and maze_quadrant == self.random_maze_quadrant
        ):
            return True
        return False

    def _check_object_task(self, env):
        """
        Check if the level contains the unseen object-task combination.

        The object color-type combination is different from the one used in self._check_color_type.

        Args:
            env: instance of the level made using a seed.

        Returns:
            bool: True if the obj.desc_fixed for a PutNext task is of the second unseen color and type, False otherwise.
        """
        for task in self._get_task_list(env):
            try:
                if (
                    task.desc_fixed.color == self.random_colors[1]
                    and task.desc_fixed.type == self.random_types[1]
                ):
                    return True
            except AttributeError:
                continue
        return False

    def _check_rel_loc(self, env):
        """
        Check if the level contains the unseen goal location description.

        This applies to Loc levels only and refers to the loc attribute of goal objects, which for most envs is None.

        Args:
            env (str): instance of the level made using a seed.

        Returns:
            bool: True if (any of) the goal object/s is/are of the unseen loc, False otherwise.
        """
        for task in self._get_task_list(env):
            try:
                if task.desc.loc and task.desc.loc == self.random_rel_loc:
                    return True
            except AttributeError:
                if (
                    task.desc_move.loc and task.desc_move.loc == self.random_rel_loc
                ) or (task.desc_fixed and task.desc_fixed.loc == self.random_rel_loc):
                    return True
        return False

    def _get_metadata_value(self, level, key, mission_space=False):
        metadata = JsoncParser.parse_file(
            os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        )["levels"]
        for levels in metadata.values():
            for l, level_info in levels.items():
                if l == level:
                    if mission_space:
                        return level_info["mission_space"][key]
                    return level_info[key]

    def _can_contain_rel_loc(self, level):
        return self._get_metadata_value(level, "location", mission_space=True)

    def _can_contain_seq(self, level):
        return self._get_metadata_value(level, "sequence", mission_space=True)

    def _can_contain_putnext(self, level):
        return self._get_metadata_value(level, "putnext")

    def _get_config_fn(self, level, config):
        level_path = os.path.join(self.seeds_dir, level)
        if not os.path.exists(level_path):
            os.mkdir(level_path)
        return os.path.join(level_path, config) + ".json"

    def load_seeds(self, level, config):
        """
        Load the seeds for all levels and configs as a dict from the json file they are stored in.

        Returns:
            dict: dictionary of dictionaries with test and validation seeds and info about the number of 'safe' train seeds.
        """
        if not os.path.exists(self._get_config_fn(level, config)):
            self.find_seeds()
        return json.load(open(self._get_config_fn(level, config), "r+"))

    def _save_seeds(self, seed_log, level, config):
        """
        Save the seeds for a given level and config from a dict to a json file.

        Args:
            seed_log (dict): dictionary of dictionaries with test and validation seeds and info about the number of 'safe' train seeds.
            level (str): name of the level.
            config (str): name of the config.
        """
        json.dump(seed_log, open(self._get_config_fn(level, config) + "_new", "w"))
        if os.path.exists(self._get_config_fn(level, config)):
            os.remove(self._get_config_fn(level, config))
        os.rename(
            self._get_config_fn(level, config) + "_new",
            self._get_config_fn(level, config),
        )

    def _create_seed_log(self, level, config):
        """
        Create a seed log for a given level and config.

        Args:
            level (str): name of the level.
            config (str): name of the config.
        """
        n_seeds_stop_search_early = 10**3

        if os.path.exists(self._get_config_fn(level, config) + "_new"):
            if os.path.exists(self._get_config_fn(level, config)):
                os.remove(self._get_config_fn(level, config))
            os.rename(
                self._get_config_fn(level, config) + "_new",
                self._get_config_fn(level, config),
            )
        if os.path.exists(self._get_config_fn(level, config)):
            try:
                seed_log = self.load_seeds(level, config)
            except Exception as e:
                print(e)
                print("Error occurred loading existing seed log, creating new one.")
                seed_log = {ood_type: {"test_seeds": []} for ood_type in self.ood_types}
                seed_log["multi_ood_seeds"] = []
                seed_log["validation_seeds"] = []
                seed_log["last_seed_tested"] = 0
                seed_log["n_train_seeds"] = 0
        else:
            seed_log = {ood_type: {"test_seeds": []} for ood_type in self.ood_types}
            seed_log["multi_ood_seeds"] = []
            seed_log["validation_seeds"] = []
            seed_log["last_seed_tested"] = 0
            seed_log["n_train_seeds"] = 0
        for seed in range(
            seed_log["last_seed_tested"],
            int(self.n_train_seeds_required * 2),
        ):
            if (
                seed_log["n_train_seeds"] == self.n_train_seeds_required
                and len(seed_log["validation_seeds"])
                == self.n_validation_seeds_required
            ):
                self._save_seeds(seed_log, level, config)
                break

            ood_type_results = {}
            for ood_type, ood_type_check in self.ood_types.items():
                if ood_type == "rel_loc" and not self._can_contain_rel_loc(level):
                    continue
                if ood_type == "object_task" and not self._can_contain_putnext(level):
                    continue
                if (
                    seed >= n_seeds_stop_search_early
                    and len(seed_log[ood_type]["test_seeds"]) == 0
                ):
                    continue
                env = gym.make(config)
                try:
                    env.reset(seed=seed)
                    _ = env.unwrapped.instrs.surface(env)
                except:
                    continue
                if ood_type_check(env):
                    if ood_type == "agent_loc" and self.ood_types["size"](env):
                        continue
                    # PutNext is not technically a maze level, but it is a special case
                    # which results in different grid dimensions, so we skip it here
                    if ood_type == "size" and config == "BabyAI-PutNextS7N4-v0":
                        continue
                    if ood_type == "agent_loc" and level == "PutNext":
                        continue
                    ood_type_results[ood_type] = True
            if len(ood_type_results) == 1:
                ood_type = list(ood_type_results.keys())[0]
                seed_log[ood_type]["test_seeds"].append(seed)
            if len(ood_type_results) > 1:
                seed_log["multi_ood_seeds"].append(seed)
            if len(ood_type_results) == 0:
                if len(seed_log["validation_seeds"]) < (
                    self.n_validation_seeds_required
                ):
                    seed_log["validation_seeds"].append(seed)
                seed_log["n_train_seeds"] += 1
            seed_log["last_seed_tested"] = seed
            if seed % 50 == 0:
                self._save_seeds(seed_log, level, config)

    def find_seeds(self, level=None):
        """
        Find and save OOD seeds for the given level and config. IID seeds are implicitly found by ruling out that a seeds is OOD.

        Seeds are OOD with respect any or all of the below:
        - size: seeds that involve an unseen room or maze size.
        - color-type: seeds that involve an unseen combination of color and type of object.
        - object-task: seeds that involve an unseen combination of object and task.
        - agent-loc: seeds that involve an unseen agent start position in the room or maze.
        - rel-loc: seeds that involve an unseen goal location relative to the agent.
        """
        threads = []
        for l, configs in self.LEVELS_CONFIGS.items():
            if level and l != level:
                continue
            for config in configs:
                print(f"Main    : create and start thread for {config}.")
                x = threading.Thread(target=self._create_seed_log, args=(l, config))
                threads.append(x)
                x.start()

        for thread in threads:
            print(f"Main    : before joining thread for {config}.")
            thread.join()
            print(f"Main    : thread for {config} done")

    def is_test_seed(self, seed_log, seed):
        """
        Check if a seed is in the list of OOD seeds for a given level and config.

        Args:
            level (str): name of the level.
            config (str): name of the config.
            seed (int): seed to check.

        Returns:
            bool: True if the seed is in the list of OOD seeds, False otherwise.

        """

        for ood_type in self.ood_types:
            if seed in seed_log[ood_type]["test_seeds"]:
                return True
        return False

    def is_validation_seed(self, seed_log, seed):
        """
        Check if a seed is in the list of validation seeds for a given level and config.

        Args:
            level (str): name of the level.
            config (str): name of the config.
            seed (int): seed to check.

        Returns:
            bool: True if the seed is in the list of validation seeds, False otherwise.
        """
        return seed in seed_log["validation_seeds"]

    def get_train_seeds(self, seed_log, level, num_train_seeds):
        if seed_log["n_train_seeds"] < num_train_seeds:
            self.n_train_seeds_required = num_train_seeds
            self.find_seeds(level)
        ceil = seed_log["last_seed_tested"] + 1
        all_seeds = np.array(range(ceil))

        exclude_seeds = seed_log["validation_seeds"] + seed_log["multi_ood_seeds"]
        for ood_type in seed_log.keys():
            if type(seed_log[ood_type]) == dict and "test_seeds" in seed_log[ood_type]:
                exclude_seeds.extend(seed_log[ood_type]["test_seeds"])
        exclude_seeds = np.array(exclude_seeds)

        return np.setdiff1d(all_seeds, exclude_seeds)
