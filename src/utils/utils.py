import os
import random
import socket
import sys
from datetime import datetime
from itertools import accumulate

import numpy as np
import torch
from minigrid.wrappers import FullyObsWrapper
from minigrid.wrappers import RGBImgObsWrapper
from minigrid.wrappers import RGBImgPartialObsWrapper
from tqdm import tqdm


def log(msg, outPath=None, with_tqdm=False):
    """Prints a message to the console and optionally writes it to a file.

    Args:
        msg (str): The message to print.
        outPath (str) (optional): The path to the file to write the message to.
    """
    msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}"

    if with_tqdm:
        tqdm.write(msg)
    else:
        print(msg)

    if outPath:
        with open(outPath, "a+") as f:
            f.write(msg + "\n")


def setup_devices(seed, useGpu=True):
    """Sets up the compute devices for training and evaluation."""
    useCuda = useGpu and torch.cuda.is_available()
    if useGpu and not useCuda:
        raise ValueError(
            "You wanted to use cuda but it is not available. "
            "Check nvidia-smi and your configuration. If you do "
            "not want to use cuda, pass the --no_gpu flag."
        )

    device = torch.device("cuda" if useCuda else "cpu")
    log(f"Using device: {torch.cuda.get_device_name()}")

    torch.manual_seed(seed)

    if useCuda:
        device_str = (
            f"{device.type}:{device.index}" if device.index else f"{device.type}"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # This does make things slower :(
        torch.backends.cudnn.benchmark = False

    return device


def is_network_connection():
    """Check if there is a working network connection."""
    host, port, timeout = "8.8.8.8", 53, 3
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        log(f"Network connection error: {ex}")
        return False


def to_one_hot(x, width=None):
    """Convert a tensor to one-hot encoding."""
    if width:
        res = np.zeros((x.size, width))
        res[np.arange(x.size), x] = 1
    else:
        res = torch.zeros_like(x)
        res[x.argmax()] = 1
    return res


def normalise(x):
    """Normalise a tensor to the range [0, 1]."""
    return (x - x.min()) / (x.max() - x.min())


def discounted_cumsum(x, gamma=1):
    """Compute the discounted cumulative sum of a tensor."""
    return np.array(list(accumulate(x[::-1], lambda a, b: (gamma * a) + b)))[::-1]


def get_minigrid_obs(env, partial_obs, fully_obs=False, rgb_obs=False):
    """
    Get the observation from the environment.

    Args:
        partial_observation (np.ndarray): the partial observation from the environment.
        env (gym.Env): the environment.

    Returns:
        np.ndarray: the observation, either as a symbolic or rgb image representation.
    """
    if fully_obs and rgb_obs:
        _env = RGBImgObsWrapper(env)
        return _env.observation({})
    elif fully_obs and not rgb_obs:
        _env = FullyObsWrapper(env)
        return _env.observation({})
    elif not fully_obs and rgb_obs:
        _env = RGBImgPartialObsWrapper(env)
        return _env.observation({})
    else:
        return partial_obs


def seed(seed):
    """Set the random seed for libraries random, numpy and pytorch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_size(num, suffix="B"):
    """Format a number of bytes to a human-readable format."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def flatten_list(l):
    """Flatten a list of lists."""
    return [item for sublist in l for item in sublist]


def frame_size(args):
    """Get the frame size used for observations."""
    if args["fully_obs"] and args["rgb_obs"]:
        return 64
    if not args["fully_obs"] and args["rgb_obs"]:
        return 56
    if args["fully_obs"] and args["rgb_obs"]:
        return 11
    return 7
