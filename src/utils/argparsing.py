import argparse

from src.constants import GLOBAL_SEED


def str2bool(v):
    """Convert a string value to a boolean, for use with argparse."""
    if isinstance(v, bool):
        return v
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Decision transformer training")

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="name of the run (default: current date and time)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10**7,
        help="the number of episodes to collect for the environment. use with policy random",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        help="the policy to use for training; can be either 'ppo' or 'random'",
    )
    parser.add_argument(
        "--include_timeout",
        type=str2bool,
        default=True,
        help="whether to include episodes terminated by timeout / truncated episodes",
    )
    parser.add_argument(
        "--fully_obs",
        type=str2bool,
        default=False,
        help="whether to use fully-observed environment",
    )
    parser.add_argument(
        "--rgb_obs",
        type=str2bool,
        default=True,
        help="whether to use rgb oberservations of the environment",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="per-device batch size for training",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=64,
        help="context length in timesteps",
    )
    parser.add_argument("--randomise_starts", type=str2bool, default=False)
    parser.add_argument(
        "--lr",
        type=float,
        default=5 * 1e-4,
        help="learning rate (default: 5 * 1e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no_gpu", action="store_true", default=False, help="disables GPU training"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="how many training steps between logging output (for PPO)",
    )
    parser.add_argument(
        "--record_video",
        type=str2bool,
        default=True,
        help="Whether to record videos of evaluation episodes",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        help="wandb mode - can be online, offline, or disabled",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="path to pytorch checkpoint file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/disk/scratch/feedback-DT/output",
        help="Path to the directory to write output to",
    )
    parser.add_argument(
        "--use_feedback",
        type=str2bool,
        default=True,
        help="whether to use feedback during training",
    )
    parser.add_argument(
        "--use_mission",
        type=str2bool,
        default=True,
        help="whether to use mission during training",
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="all",
        help="which type of feedback to use during training; can be either 'all', 'rule', 'task', 'numerical' or 'random'",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="GoToRedBallGrey",
        help="the name of the level to train on",
    )
    parser.add_argument(
        "--eval_step_interval",
        type=int,
        default=5,
        help="after how many steps to evaluate the sample efficiency of the model.",
    )
    parser.add_argument(
        "--target_return",
        type=int,
        default=1,
        help="the target return to condition on",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=128,
        help="number of seeds to evaluate over",
    )
    parser.add_argument(
        "--load_existing_dataset",
        type=str2bool,
        default=False,
        help="whether to load the dataset if it already exists",
    )
    parser.add_argument(
        "--eps_per_shard",
        type=int,
        default=4,
        help="the number of episodes to collect per dataset shard. this will be 100 for simple tasks.",
    )
    parser.add_argument(
        "--use_full_ep",
        type=str2bool,
        default=False,
        help="whether to use the full episode, rather than a given context length",
    )
    parser.add_argument(
        "--ep_dist",
        type=str,
        default="uniform",
        help="the distribution from which to sample episodes; can be 'uniform', 'inverse', or 'length",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="how many training steps between logging output (for HF Trainer)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=8,
        help="how many eval intervals to wait for improvements in the evaluation metric before stopping training, should be twice as long for multi-room tasks, and incerase/decrease proportional to sample interval if this is not the default (default 20 -> 512)",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.001,
        help="the threshold by which improvements in the evaluation metric have to exceed the previous best performance for early stopping",
    )
    parser.add_argument(
        "--ppo_early_stopping_patience",
        type=int,
        default=20,
        help="how many steps to wait for improvements in the evaluation metric before stopping training",
    )
    parser.add_argument(
        "--ppo_early_stopping_threshold",
        type=float,
        default=0.02,
        help="the threshold by which improvements in the evaluation metric have to exceed the previous best performance for early stopping",
    )
    parser.add_argument(
        "--use_rtg",
        type=str2bool,
        default=True,
        help="whether or not to condition on the RTG",
    )
    parser.add_argument(
        "--eps_per_seed",
        type=int,
        default=10,
        help="how many episodes per seed to generate. use with policy random",
    )
    parser.add_argument(
        "--num_train_seeds",
        type=int,
        default=128,
        help="how many training seeds to generate episodes from. use with policy random",
    )
    parser.add_argument(
        "--mission_at_inference",
        type=str,
        default="actual",
        help="representation to use for mission at inference time; can be either 'numerical', 'string' or 'actual'",
    )
    parser.add_argument(
        "--feedback_at_inference",
        type=str,
        default="numerical",
        help="representation to use for feedback at inference time; can be either 'numerical', 'zero', 'mean', 'string' or 'actual'",
    )
    parser.add_argument(
        "--mission_mode",
        type=str,
        default="standard",
        help="which type of feedback to use during training; can be either 'standard' or 'random'",
    )
    parser.add_argument(
        "--random_mode",
        type=str,
        default="english",
        help="which type of feedback to use during training; can be either 'english' or 'lorem'",
    )
    parser.add_argument(
        "--loss_mean_type",
        type=str,
        default="ce",
        help="how to form the mean loss; can be either 'ce' or 'custom'",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=GLOBAL_SEED,
        help="whether to use the feedback loss",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=987654321,
        help="the seed used for seeding different instantiations of the model",
    )
    parser.add_argument(
        "--demo_config",
        type=str,
        default="BabyAI-GoToRedBallGrey-v0",
        help="the config used for instantiating the env for the demo",
    )
    parser.add_argument(
        "--demo_seed",
        type=int,
        default=0,
        help="the seed used for seeding the env for the demo",
    )
    parser.add_argument(
        "--analyse_only_level",
        type=str,
        default="",
        help="single level on whose results to run data analysis",
    )
    parser.add_argument(
        "--analysis_metric",
        type=str,
        default="gc_success",
        help="metric to use for data analysis",
    )
    parser.add_argument(
        "--analysis_fig_size",
        type=str,
        default="small-half",
        help="figure size for data analysis; follows format 'small', 'small-half, 'small-third', 'medium', ..., 'large-third'",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="conditioning",
        help="experiment name; used for specifying the path to the data",
    )

    return vars(parser.parse_args())
