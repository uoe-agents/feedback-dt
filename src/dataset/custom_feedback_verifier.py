import re
from abc import ABC
from abc import abstractmethod

from essential_generators import DocumentGenerator
from lorem_text import lorem
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Box
from minigrid.core.world_object import Door
from minigrid.core.world_object import Key
from minigrid.core.world_object import Wall
from minigrid.envs.babyai.core.verifier import ActionInstr
from minigrid.envs.babyai.core.verifier import AfterInstr
from minigrid.envs.babyai.core.verifier import AndInstr
from minigrid.envs.babyai.core.verifier import BeforeInstr
from minigrid.envs.babyai.core.verifier import GoToInstr
from minigrid.envs.babyai.core.verifier import LOC_NAMES
from minigrid.envs.babyai.core.verifier import OBJ_TYPES
from minigrid.envs.babyai.core.verifier import ObjDesc
from minigrid.envs.babyai.core.verifier import OpenInstr
from minigrid.envs.babyai.core.verifier import PickupInstr
from minigrid.envs.babyai.core.verifier import pos_next_to
from minigrid.envs.babyai.core.verifier import PutNextInstr
from minigrid.envs.babyai.core.verifier import SeqInstr

SEQUENCE_CONSTRUCTORS = ["and", ", then", "after you"]
ACTION_WORDS = ["go to", "open", "pick up", "put"]


class GoNextToInstr(ActionInstr):
    """
    Go next to an object while carrying one.
    eg: go next to the red ball.
    Used as a subgoal / to generate feedback for put next to mission.

    Args:
        obj_move (ObjDesc): Object to move next to.
        obj_fixed (ObjDesc): Object to move next to obj_move.
    """

    def __init__(self, obj_move, obj_fixed):
        super().__init__()
        self.desc_move = obj_move
        self.desc_fixed = obj_fixed

    def surface(self, env):
        """
        Return a surface representation of the instruction.

        Args:
            env (Env): Environment containing the objects.
        """
        return (
            "put "
            + self.desc_move.surface(env)
            + " next to "
            + self.desc_fixed.surface(env)
        )

    def reset_verifier(self, env):
        """
        Reset the feedback verifier.

        Args:
            env (Env): Environment.
        """
        super().reset_verifier(env)
        self.desc_move.find_matching_objs(env)
        self.desc_fixed.find_matching_objs(env)
        self.front_pos = env.front_pos
        self.front_cell = env.grid.get(*self.front_pos)

    def verify_action(self, action):
        """
        Verify whether the action taken by the agent was successful.

        Args:
            action (int): The action to verify.

        Returns:
            str: The verification result.
        """
        if action in [
            self.env.actions.drop,
            self.env.actions.pickup,
            self.env.actions.toggle,
            self.env.actions.done,
        ]:
            return "continue"

        if self.env.carrying not in self.desc_move.obj_set:
            return "continue"

        pos_a = self.env.front_pos

        if self.front_cell is not None:
            return "continue"

        for pos_b in self.desc_fixed.obj_poss:
            if pos_next_to(pos_a, pos_b):
                return "success"

        return "continue"


class Feedback(ABC):
    """
    Super class for generating feedback for actions on BabyAI environments.
    """

    env = None
    action = None
    front_pos = None
    front_cell = None
    carrying = None

    @abstractmethod
    def verify_feedback(self, env, action):
        """
        Verify the feedback for the action taken by the agent.

        Args:
            env (MiniGridEnv): The environment which to verify an action against. MiniGridEnv is a subclass of gym.Env.
            action (int): The action to verify.
        """
        raise NotImplementedError

    def _is_empty_cell(self):
        """
        Check if the agent is positioned in front of an empty cell.

        Returns:
            bool: True if the agent is positioned in front of an empty cell, False otherwise.
        """
        return self.front_cell is None

    def _is_wall(self):
        """
        Check if the agent is positioned in front of a wall.

        Returns:
            bool: True if the agent is positioned in front of a wall, False otherwise.
        """
        return isinstance(self.front_cell, Wall)

    def _is_door(self):
        """
        Check if the agent is positioned in front of a door.

        Returns:
            bool: True if the agent is positioned in front of a door, False otherwise.
        """
        return isinstance(self.front_cell, Door)

    def _is_open_door(self):
        """
        Check if the agent is positioned in front of an open door.

        Returns:
            bool: True if the agent is positioned in front of an open door, False otherwise.
        """

        return self._is_door() and self.front_cell.is_open


class RuleFeedback(Feedback):
    """
    Sub class for generating rule feedback for actions on BabyAI environments.
    """

    def _is_obstacle(self):
        """
        Check if there is an obstacle object in front of the agent.

        Returns:
            bool: True if the object in front of the agent is an obstacle (other than a closed/locked door or wall), False otherwise.
        """
        return not self.front_cell.can_overlap() and not (
            self._is_closed_door() or self._is_locked_door() or self._is_wall()
        )

    def _is_closed_door(self):
        """
        Check if the agent is positioned in front of a closed door.

        Returns:
            bool: True if the agent is positioned in front of a closed door, False otherwise.
        """
        if self._is_door():
            return not self.front_cell.is_open and not self.front_cell.is_locked
        return False

    def _is_locked_door(self):
        """
        Check if the agent is positioned in front of a locked door.

        Returns:
            bool: True if the agent is positioned in front of a locked door, False otherwise.
        """
        if self._is_door():
            return self.front_cell.is_locked
        return False

    def _is_box(self):
        """
        Check if there is a box object in front of the agent.

        Returns:
            bool: True if the object in front of the agent is a box, False otherwise.
        """
        if isinstance(self.front_cell, Box):
            return True
        return False

    def _is_carrying(self):
        """
        Check if the agent is carrying an object.

        Returns:
            bool: True if the agent is carrying an object, False otherwise.
        """
        return self.carrying is not None

    def _is_carrying_key(self):
        """
        Check if the agent is carrying a key.

        Returns:
            bool: True if the agent is carrying a key, False otherwise.
        """
        return isinstance(self.carrying, Key)

    def _is_carrying_correct_key(self):
        """
        Check if the agent is carrying a correct key to unlock the door it is positioned in front of.

        Returns:
            bool: True if the agent is carrying a correct key to unlock the door it is positioned in front of, False otherwise.
        """
        return self._is_carrying_key() and self.carrying.color == self.front_cell.color

    def _is_valid_move_forward(self):
        """
        Check if the agent can move forward.

        Returns:
            bool: True if the agent can move forward, False otherwise.
        """
        return self._is_empty_cell() or self._is_open_door()

    def _get_move_forward_feedback(self):
        """
        Return the feedback for the move forward action.

        Returns:
            str: The feedback for the move forward action with respect to the object in the cell that the agent is facing.
        """
        if self._is_locked_door():
            return "Not a good idea! You can't move forward here as the door in front of you is locked."
        if self._is_closed_door():
            return "Not a good idea! You can't move forward here as the door in front of you is closed."
        if self._is_wall():
            return (
                "Not a good idea! You can't move forward while you're facing the wall."
            )
        if self._is_obstacle():
            return (
                "Not a good idea! You can't move forward here "
                + f"as there's a {self.front_cell.type} blocking the way."
            )
        return "No feedback available."

    def _is_valid_toggle(self):
        """
        Check if the agent can toggle the object in front of it.

        Returns:
            bool: True if the agent can toggle the object in front of it, False otherwise.
        """
        if self.front_cell:
            return (
                (self._is_locked_door() and self._is_carrying_correct_key())
                or self._is_closed_door()
                or self._is_box()
            )
        return False

    def _get_toggle_feedback(self):
        """
        Return the feedback for the toggle action.

        Returns:
            str: The feedback for the toggle action with respect to the object in the cell that the agent is facing.
        """
        if self._is_empty_cell():
            return "That won't work here. There's nothing in front of you, and you can't open empty space."
        if (
            self._is_locked_door()
            and self._is_carrying_key()
            and not self._is_carrying_correct_key()
        ):
            return f"That won't work here. You can't open a locked door without a key of the same color as the door. You're carrying a {self.carrying.color} key, but the door in front of you is {self.front_cell.color}."
        if self._is_locked_door() and not self._is_carrying_key():
            return "That won't work here. You can't open a locked door without a key of the same color as the door, and you're not carrying any key."
        if self._is_wall():
            return "That won't work here. You can't open the wall."
        if self._is_obstacle():
            return f"That won't work here. You can't open {self.front_cell.type}s."
        return "No feedback available."

    def _is_valid_pickup(self):
        """
        Check if the agent can pick up the object in front of it.

        Returns:
            bool: True if the agent can pick up the object in front of it, False otherwise.
        """
        if self.front_cell and not self.carrying:
            return self.front_cell.can_pickup()
        return False

    def _get_pickup_feedback(self):
        """
        Return the feedback for the pickup action.

        Returns:
            str: The feedback for the pickup action with respect to the object in the cell that the agent is facing.
        """
        if self._is_empty_cell():
            return "Not a good idea! There's nothing in front of you, and you can't pick up empty space."
        if self._is_door():
            return "Not a good idea! You can't pick up doors."
        if self._is_wall():
            return "Not a good idea! You can't pick up the wall."
        if self._is_carrying():
            return "Not a good idea! You can't pick up another object while you're already carrying one."
        return "No feedback available."

    def _is_valid_drop(self):
        """
        Check if the agent can drop an object it is carrying.

        Returns:
            bool: True if the agent can drop an object it is carrying, False otherwise.
        """
        return self._is_carrying() and self._is_empty_cell()

    def _get_drop_feedback(self):
        """
        Return the feedback for the drop action.

        Returns:
            str: The feedback for the drop action with respect to the object the agent is carrying and the cell that the agent is facing.
        """
        if not self._is_carrying():
            return "Don't do that! You're not carrying any object so dropping has no effect."
        if self._is_wall():
            return (
                "Don't do that! You can't drop an object while you're facing the wall."
            )
        if self._is_door():
            return "Don't do that! You can't drop an object while you're facing a door."
        if self._is_obstacle():
            return (
                "Don't do that! You can't drop an object on top of another object, and "
                + f"there's already a {self.front_cell.type} in front of you."
            )
        return "No feedback available."

    def _get_rule_feedback(self):
        """
        Return the rule violation feedback for the action taken by the agent.

        Returns:
        str
            The feedback for the action taken by the agent.
        """
        if (
            self.action == self.env.actions.forward
            and not self._is_valid_move_forward()
        ):
            return self._get_move_forward_feedback()
        if self.action == self.env.actions.toggle and not self._is_valid_toggle():
            return self._get_toggle_feedback()
        if self.action == self.env.actions.pickup and not self._is_valid_pickup():
            return self._get_pickup_feedback()
        if self.action == self.env.actions.drop and not self._is_valid_drop():
            return self._get_drop_feedback()
        return "No feedback available."

    def verify_feedback(self, env, action):
        """
        Verify the feedback for the action taken by the agent.

        Raises
        ------
        NotImplementedError
            Raised when not overriden by a derived class
        """
        self.env = env
        self.front_pos = self.env.front_pos
        self.front_cell = self.env.unwrapped.grid.get(*self.front_pos)
        self.carrying = self.env.carrying
        self.action = action

        return self._get_rule_feedback()


class TaskFeedback(Feedback):
    """
    Sub class for generating task feedback for actions on BabyAI environments.
    """

    def __init__(self, env, test_mode=False):
        self.env = env
        self.tasks = self._get_tasks()
        self.subtasks = self._get_subtasks()
        self.agent_pos = self.env.unwrapped.agent_pos
        if test_mode:
            self.pop_from = -1
        else:
            self.pop_from = 0

    # METHODS FOR DECOMPOSING TASKS INTO SUBTASKS

    def _task_is_sequence(self):
        """Check if the task is a sequence type task."""
        return isinstance(self.env.unwrapped.instrs, SeqInstr)

    # Instructions for AfterInst are sequences linked by inst_a 'after you' inst_b
    def _task_is_after(self):
        """Check if the task is an 'after' type task."""
        return isinstance(self.env.unwrapped.instrs, AfterInstr)

    # Instructions for BeforeInst are sequences linked by inst_a ', then' inst_b
    def _task_is_before(self):
        """Check if the task is a 'before' type task."""
        return isinstance(self.env.unwrapped.instrs, BeforeInstr)

    def _task_is_and(self, instrs):
        """Check if the task is an 'and' type task."""
        return isinstance(instrs, AndInstr)

    def _task_is_goto(self, instrs):
        """Check if the task is a 'go to' type task."""
        return isinstance(instrs, GoToInstr)

    def _task_is_go_next_to(self, instrs):
        """Check if the task is a 'go next to' type task."""
        return isinstance(instrs, GoNextToInstr)

    def _task_is_open(self, instrs):
        """Check if the task is an 'open' type task."""
        return isinstance(instrs, OpenInstr)

    def _task_is_unlock(self, instrs):
        """Check if the task is an 'unlock' type task."""
        door_pos = instrs.desc.obj_poss[0]
        door = self.env.unwrapped.grid.get(*door_pos)
        return self._task_is_open(instrs) and door.is_locked

    def _task_is_pickup(self, instrs):
        """Check if the task is a 'pickup' type task."""
        return isinstance(instrs, PickupInstr)

    def _task_is_putnext(self, instrs):
        """Check if the task is a 'put next' type task."""
        return isinstance(instrs, PutNextInstr)

    # THIS DECIDES THE ORDER IN WHICH FEEDBACK IS PROVIDED, HOWEVER THE ORDER OF
    # 'AND' SUBTASKS SHOULD BE ALLOWED TO BE ARBITRARY
    def _decompose_and_instrs(self, instrs):
        """Decompose an 'and' type task into its subtasks."""
        if self._task_is_and(instrs):
            return instrs.instr_a, instrs.instr_b
        return [instrs]

    def _get_tasks(self):
        """Get the task instructions for the current environment."""
        if self._task_is_before():
            return [
                *self._decompose_and_instrs(self.env.unwrapped.instrs.instr_a),
                *self._decompose_and_instrs(self.env.unwrapped.instrs.instr_b),
            ]
        if self._task_is_after():
            return [
                *self._decompose_and_instrs(self.env.unwrapped.instrs.instr_b),
                *self._decompose_and_instrs(self.env.unwrapped.instrs.instr_a),
            ]
        if self._task_is_and(self.env.unwrapped.instrs):
            return [*self._decompose_and_instrs(self.env.unwrapped.instrs)]
        return [self.env.unwrapped.instrs]

    def _decompose_open_instrs(self, instrs):
        """Decompose an 'open' type task into its subtasks."""
        return GoToInstr(instrs.desc), instrs

    def _decompose_unlock_instrs(self, instrs):
        """Decompose an 'unlock' type task into its subtasks."""
        goto_key_instrs = GoToInstr(ObjDesc("key", instrs.desc.color))
        goto_key_instrs.reset_verifier(self.env)
        pickup_key_instrs = PickupInstr(ObjDesc("key", instrs.desc.color))
        pickup_key_instrs.reset_verifier(self.env)
        return (
            goto_key_instrs,
            pickup_key_instrs,
            GoToInstr(instrs.desc),
            instrs,
        )

    def _decompose_pickup_instrs(self, instrs):
        """Decompose a 'pickup' type task into its subtasks."""
        return GoToInstr(instrs.desc), instrs

    def _decompose_putnext_instrs(self, instrs):
        """Decompose a 'put next' type task into its subtasks."""
        return (
            GoToInstr(instrs.desc_move),
            PickupInstr(instrs.desc_move),
            GoNextToInstr(instrs.desc_move, instrs.desc_fixed),
            instrs,
        )

    def _get_subtasks(self):
        """Get the subtasks for the current environment."""
        subtasks = []
        for task in self.tasks:
            if self._task_is_goto(task):
                subtasks.append(task)
            if self._task_is_open(task):
                if self._task_is_unlock(task):
                    subtasks.extend(self._decompose_unlock_instrs(task))
                else:
                    subtasks.extend(self._decompose_open_instrs(task))
            if self._task_is_pickup(task):
                subtasks.extend(self._decompose_pickup_instrs(task))
            if self._task_is_putnext(task):
                subtasks.extend(self._decompose_putnext_instrs(task))
        return subtasks

    # METHODS FOR GENERATING FEEDBACK FOR EACH SUBTASK

    def _is_goal(self, current_obj, goal_obj):
        """Check if the current object is the goal object."""
        return current_obj in goal_obj.obj_set

    def _is_next_to_goal(self, goal_poss, current_pos):
        """Check if the agent is next to the goal object."""
        for pos in goal_poss:
            if pos_next_to(pos, current_pos):
                return True
        return False

    def _has_multiple_goals(self, goal_obj):
        """Check if there are multiple goal objects."""
        return len(goal_obj.obj_set) > 1

    def _get_article(self, goal_obj):
        """Get the article to use for the goal object."""
        if self._has_multiple_goals(goal_obj):
            return "a"
        return "the"

    def _get_completion_level(self):
        """Get the completion level of the task."""
        if not self.subtasks:
            return ""
        return "a part of "

    def _get_object_string(self, obj):
        """Get a natural language description of the object."""
        if obj.loc:
            location_string = (
                f"on your {obj.loc}"
                if obj.loc in ["left", "right"]
                else f"on your {obj.loc}"
            )
        else:
            location_string = ""
        object_string = f"{self._get_article(obj)} {obj.color} {obj.type}"
        object_string += f" {location_string}" if location_string else ""
        return object_string

    def _get_goto_feedback(self, instrs):
        """Get feedback for a 'go to' type task."""
        goal_obj = instrs.desc
        if not (self._is_wall() or self._is_empty_cell()):
            if self._is_goal(self.front_cell, goal_obj):
                self.subtasks.pop(self.pop_from)
                return f"Fantastic! You've completed {self._get_completion_level()}your task by going to {self._get_object_string(goal_obj)}."
        return "No feedback available."

    def _get_open_feedback(self, instrs):
        """Get feedback for an 'open' type task."""
        goal_obj = instrs.desc
        if not (self._is_wall() or self._is_empty_cell()):
            if self._is_goal(self.front_cell, goal_obj):
                if self._is_open_door():
                    self.subtasks.pop(self.pop_from)
                    return f"That's correct! You've completed {self._get_completion_level()}your task by opening {self._get_object_string(goal_obj)}."
        return "No feedback available."

    def _get_pickup_feedback(self, instrs):
        """Get feedback for a 'pickup' type task."""
        goal_obj = instrs.desc
        if self._is_goal(self.carrying, goal_obj):
            self.subtasks.pop(self.pop_from)
            return f"Great job! You've completed {self._get_completion_level()}your task by picking up {self._get_object_string(goal_obj)}."
        return "No feedback available."

    def _get_go_next_to_feedback(self, instrs):
        """Get feedback for a 'go next to' type task."""
        goal_obj_1 = instrs.desc_move
        goal_obj_2 = instrs.desc_fixed
        if self._is_goal(self.carrying, goal_obj_1):
            if (
                self._is_next_to_goal(goal_obj_2.obj_poss, self.front_pos)
                and self._is_empty_cell()
            ):
                self.subtasks.pop(self.pop_from)
                return f"That's right! You've completed {self._get_completion_level()}your task by going next to {self._get_object_string(goal_obj_2)}."
        return "No feedback available."

    def _get_putnext_feedback(self, instrs):
        """Get feedback for a 'put next' type task."""
        goal_obj_1 = instrs.desc_move
        goal_obj_2 = instrs.desc_fixed
        if self._is_goal(self.front_cell, goal_obj_1):
            if self._is_next_to_goal(goal_obj_2.obj_poss, self.front_pos):
                self.subtasks.pop(self.pop_from)
                return f"Well done! You've completed {self._get_completion_level()}your task by putting {self._get_object_string(goal_obj_1)} next to {self._get_object_string(goal_obj_2)}."
        return "No feedback available."

    def _get_task_feedback(self):
        """Get task feedback for the action taken by the agent."""
        try:
            current_subtask = self.subtasks[self.pop_from]
        except IndexError:
            return "No feedback available."
        if (
            self.action == self.env.actions.left
            or self.action == self.env.actions.right
            or self.action == self.env.actions.forward
        ):
            if self._task_is_goto(current_subtask):
                return self._get_goto_feedback(current_subtask)
            if self._task_is_go_next_to(current_subtask):
                return self._get_go_next_to_feedback(current_subtask)
        if self.action == self.env.actions.toggle and self._task_is_open(
            current_subtask
        ):
            return self._get_open_feedback(current_subtask)
        if self.action == self.env.actions.pickup and self._task_is_pickup(
            current_subtask
        ):
            return self._get_pickup_feedback(current_subtask)
        if self.action == self.env.actions.drop and self._task_is_putnext(
            current_subtask
        ):
            return self._get_putnext_feedback(current_subtask)
        return "No feedback available."

    def verify_feedback(self, env, action):
        self.env = env
        self.action = action
        self.front_pos = self.env.front_pos
        self.front_cell = self.env.unwrapped.grid.get(*self.front_pos)
        self.carrying = self.env.carrying

        return self._get_task_feedback()


class RandomFeedback:
    """
    Class for generating random feedback (for ablations)
    """

    def __init__(self, random_type):
        self.random_type = random_type
        self.babyai_words = (
            OBJ_TYPES + LOC_NAMES + COLOR_NAMES + ACTION_WORDS + SEQUENCE_CONSTRUCTORS
        )

    def get_random_sentences(self):
        """
        Get random feedback to replace actual feedback with (for ablations).

        Args:
            random_type (str): The type of random feedback to generate. Can be either 'english' or 'lorem_ipsum'.

        Returns:
            str: The random feedback.
        """
        if self.random_type == "english":
            generator = DocumentGenerator()
            babyai_words = (
                OBJ_TYPES
                + LOC_NAMES
                + COLOR_NAMES
                + ACTION_WORDS
                + SEQUENCE_CONSTRUCTORS
            )
        sentences = []

        while len(sentences) < 100:
            if "lorem" in self.random_type:
                sentence = lorem.sentence()
                while len(sentence) > 150:
                    sentence = lorem.sentence()
                sentences.append(sentence)
            else:
                word_list = babyai_words
                while any(word in babyai_words for word in word_list):
                    sentence = generator.sentence()
                    word_list = set(re.sub(r"\W+", " ", sentence).lower().split())
                sentences.append(sentence)

        return sentences
