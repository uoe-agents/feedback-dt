import os
import shutil
from pathlib import Path


def get_file_path(dataset_name):
    datasets_path = os.environ.get("MINARI_DATASETS_PATH")
    if datasets_path is not None:
        file_path = os.path.join(datasets_path, f"{dataset_name}.hdf5")
    else:
        datasets_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
        file_path = os.path.join(datasets_path, f"{dataset_name}.hdf5")

    os.makedirs(datasets_path, exist_ok=True)
    return Path(file_path)


def list_local_datasets():
    datasets_path = get_file_path("").parent
    return [
        f[:-5]
        for f in os.listdir(datasets_path)
        if os.path.isfile(os.path.join(datasets_path, f))
    ]


def name_dataset(args):
    level = args['level']
    obs = f"_obs-{'rgb' if args['rgb_obs'] else 'symb'}"
    size = f"_size-n-seeds-{args['num_train_seeds']}_eps-per-seed-{args['eps_per_seed']}" if args["policy"] == "random" else ""
    feedback_mode = f"_feedback-{args['feedback_mode']}"
    print(size)
    return (
        level
        + "_policy-"
        + args["policy"]
        + obs
        + size
        + feedback_mode
    )


def delete_dataset(dataset_name):
    dataset_path = get_file_path(dataset_name)
    shutil.rmtree(dataset_path)
    print(f"Sucess! Deleted {dataset_name}")
