# feedback-DT

This repository houses all the code for our paper '*Is Feedback All You Need? Leveraging Natural Language Feedback in Goal-Conditioned Reinforcement Learning*', as well as ongoing work developing the project further (for the specific version of the code matching the workshop paper, see the `neurips2023_gcrl` branch). If you use any of the code for your own work, please cite the paper as
```
@inproceedings{
        mccallum2023feedback,
        title={Is Feedback All You Need? Leveraging Natural Language Feedback in Goal-Conditioned Reinforcement Learning},
        author={Sabrina McCallum and Max Taylor-Davies and Stefano V. Albrecht and Alessandro Suglia},
        booktitle={NeurIPS Workshop on Goal-Conditioned Reinforcement Learning (GCRL)},
        year={2023},
    }
```

# Overview and usage
The main purpose of this codebase is to train and evaluate feedback-augmented decision-transformer agents on an extended version of the [BabyAI testbed](https://github.com/mila-iqia/babyai/tree/master).

## Installation
You can clone the repository and install dependencies with
```bash
git clone https://github.com/maxtaylordavies/feedback-DT.git
cd feedback-DT
pipenv install
```

## Training and evaluating an FDT agent
The script `train_agent_babyai.py` contains everything needed to
- generate a training dataset
- train an FDT agent
- run periodic evaluation during training and save results

To run it with default arguments, simply execute
```bash
python scripts/train_agent_babyai.py
```

The script also supports a number of arguments. The main ones are highlighted below; for a complete list refer to `src/utils/argparsing.py`.

### Specifying a BabyAI level
The `--level` argument determines which BabyAI level to use, given as the middle part of the config name, e.g. "BabyAI-GoToRedBallGrey-v0" -> "GoToRedBallGrey" (the default value). A list of levels can be found on [this page](https://minigrid.farama.org/environments/babyai/) under "Registered Configurations".

### Controlling dataset creation
To avoid creating a new dataset everytime the script is run, you can use the `--load_existing_dataset` argument - this will check if a dataset matching all the relevant arguments exists, and load it if so. To download datasets created for the experiments in the paper, go to [this dropbox URL](https://www.dropbox.com/sh/b0bff46d4s230hr/AADJE4xu_Aliqd_mAv3kUdSda?dl=0).

By default, datasets are generated using a random policy - to instead sample data from the training process of a PPO agent, set `--policy` to "ppo". If using a PPO policy, `--num_steps` controls the number of environment steps to include in the training dataset; for a random policy you can set `--num_train_seeds` and `--eps_per_seed` to determine the number of episodes created.

Finally, to exclude episodes that end as a result of timing out, set `--include_timeout` to false.

### Controlling what action prediction is conditioned on
The original decision transformer uses return-to-go (RTG) to condition action predictions during training. Our architecture allows conditioning on RTG, mission string, language feedback, or any combination - you can control this by setting the `--use_rtg`, `--use_mission` and `--use_feedback` arguments. You can also control the provision of feedback and mission strings at *inference* time using the `--feedback_at_inference` and `--mission_at_inference` arguments.

### Controlling feedback generation
If conditioning on feedback (see previous section), you can determine which *type* of feedback is used by setting the `--feedback_mode` argument to one of 'all', 'rule', 'task', 'numerical' or 'random'. In numerical mode, feedback is just given as a "1" when task feedback is available, "-1" when rule feedback is available, and "0" otherwise. If using random feedback, you can additionally specify the type of random feedback through `--random_mode`, which can be either 'english' (uses random nonsensical English sentences that exclude words in the BabyAI vocabulary) or 'lorem' (uses lorem ipsum text). For an explanation of what the other feedback types mean, see section 3 and appendix A.1 of the paper.

### Controlling evaluation
Tthe training process includes automatic evaluation of the agent's  performance at periodic intervals, relative to a random policy baseline. By default, evaluation runs every 5 training steps; you can customise this by setting `--eval_step_interval`. At each evaluation, performance is averaged over a subset of environment seeds contained within the training dataset. You can set the number of seeds used with `--num_repeats`. At the end of training, we run an additional evaluation of the agent's in-distribution and out-of-distribution generalisation performance - `--num_repeats` also determines the number of seeds used here.

The results from the periodic evaluations are also used to determine whether to stop training early - to customise this, set the `--early_stopping_patience` and `--early_stopping_threshold` arguments.

# Architecture
This section is intended as an overview of the codebase architecture for anyone wishing to modify or extend the code for their own purposes. There are 6 main components to the architecture: `Agent`, `FeedbackEnv`, `Dataset`, `Collator`, `AgentTrainer`, `Evaluator`

### Agent
The `Agent` class basically represents any offline-RL-ish agent; i.e. something that we want to train from a dataset of sequential environment interactions, and then use as a policy to produce actions given observations. This is what its basic template definition looks like:
```python
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
        pass

    def _compute_loss(self, input: AgentInput, output: Any) -> float:
        pass

    def forward(self, input: AgentInput, **kwargs) -> Dict:
        output = self._forward(input)
        loss = self._compute_loss(input, output)
        return {"loss": loss}

    def get_action(self, input: AgentInput, context=1, one_hot=False):
        pass
```

Implementing any particular type of agent/model then mostly consists of overriding the `_forward`, `_compute_loss` and `get_action` methods.

The file `agent/fdt/base.py` contains an `Agent` implementation for the feedback decision transformer. This is the file that contains almost all the relevant logic for the actual FDT model. The file `agent/fdt/minigrid.py` contain the version of the agent for use with Minigrid/BabyAI environments. In the future, we may add additional subclasses to `agent/fdt/` for other types of environment.

### FeedbackEnv
The `FeedbackEnv` class (`env/feedback_env.py`) is a wrapper around a gymnasium `Env` that adds the ability to generate and return natural language feedback on each step. To create a `FeedbackEnv` instance, you should first instantiate a regular `Env`, and then pass it in along with the feedback mode and max environment steps:
```python
_env = gym.make(env_key, render_mode=render_mode)
 _env.reset(seed=seed)
env = FeedbackEnv(_env, feedback_mode=feedback_mode, max_steps=max_steps)
```
Once you have a `FeedbackEnv`, you can mostly use it as you would a regular `Env`. The main difference form a usage perspective is that the fifth element of the tuple returned by a call to `step` is now a string (possibly) containing feedback, rather than an info dict.


### Dataset
The `CustomDataset` class (`dataset/custom_dataset.py`) essentially represents an interface to a collection of `hdf5` files on disk, where each file contains some number of recorded steps. Each step consists of the elements (mission, observation, action, reward, feedback, truncated, terminated). A new `CustomDataset` can be created by sampling from some specified policy (or randomly). Once a `CustomDataset` has been created and saved to disk (across some number of sharded files), we can sample data from it using the following three methods:

### Collator
The `Collator` (`src/collator/collator.py`) basically functions as an intermediary between a `CustomDataset` and an `Agent`, for the purposes of training: it samples batches of data from the `CustomDataset`, and packages them as inputs to be processed by the `Agent`. To create a collator, you pass in a dataset object and a bunch of optional args:
```python
def __init__(
        self,
        custom_dataset: CustomDataset,
        feedback=True,
        mission=True,
        context_length=64,
        scale=1,
        gamma=0.99,
        embedding_dim=128,
        randomise_starts=False,
        episode_dist="uniform",
    ) -> None:
```

The collator also handles the responsibility of embedding the natural language data (mission strings, feedback strings). This lets us swap out different embeddings models without having to modify the agent architecture. Since we expect a lot of repetition in the values of these strings, we implement a memoised embedder: the collator maintains a local embeddings cache; each time we get a mission or feedback string to embed, we check if we already have an embedding for that cache in the string, and only run the embedding model if not.

The collator's main method is `_sample_batch` - this samples a batch of sub-episodes from the dataset, does some processing (embedding strings and padding everything to context length), and then returns a dict with these keys:
```python
{
    "timesteps": ...,
    "mission_embeddings": ...,
    "states": ...,
    "actions": ...,
    "rewards": ...,
    "returns_to_go": ...,
    "attention_mask": ...,
    "feedback_embeddings": ...,
}
```

### Trainer
Our `AgentTrainer` class is just a very simple wrapper around the HuggingFace transformers `Trainer`, that works with our other components and handles a bunch of default arguments and settings, as well as automatic creation of our evaluation callbacks. Once you have an agent, dataset and collator, you create it like this:
```python
trainer = AgentTrainer(
    agent=agent,
    collator=collator,
    dataset=dataset,
    args=args,
)
```
and then run it with
```python
trainer.train()
```

### Evaluator
Our `Evaluator` class is where we implement the evaluation functions that are run on the agent during training. It subclasses the HuggingFace `TrainerCallback` class. This lets us write methods that will e.g. be called at the end of every training step:
```python
def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Agent,
        **kwargs,
    ):
        self._plot_loss(state)

        if state.global_step % self.eval_step_interval == 0 and state.global_step > 0:
            self._run_eval_and_plot(model, state, eval_type="efficiency", control=control)
            model.save_checkpoint(self.output_dir, state.global_step)
```
What happens here is that we first update our plot of the training loss (this is saved to disk at each step), and then check whether the current step is a multiple of `self.eval_step_interval` (determined by the `--eval_step_interval` command line argument) - if it is, we call `_run_eval_and_plot`. This evaluates the performance of the current agent, and updates the eval plot(s) saved on disk. Specifically: we sample *n* instances of the environment; for each of those, we sample a trajectory from the agent being trained, and also from a random agent (which serves as a baseline). For both trajectories we record the return, the length, and whether it was successful. All three of these metrics are plotted throughout training against number of samples. To track any additional metrics, you can just add code to `_run_eval_and_plot`.

## Exploring feedback
We provide a wrapper around Minigrid's `ManualControl` which provides a UI for human users to play BabyAI levels. Our wrapper saves each frame, a gif of the episode and a log file to which we write the mission and the feedback provided at each step. The corresponding make_demo.py script can be used to generate and save demo videos and for a specific `--demo_config` and `--demo_seed` and explore the feedback generated from interactions with the environment.
