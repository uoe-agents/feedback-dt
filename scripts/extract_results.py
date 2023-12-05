from src.utils.argparsing import get_args
import os
import shutil

USER = os.environ["USER"]
ROOT = os.path.dirname(os.path.abspath(USER))
args = get_args()
experiment_name = args["experiment_name"]
data_home = f"{ROOT}/data/{experiment_name}/output"

results_home = f"{ROOT}/data/{experiment_name}/output-results-only"

if not os.path.exists(results_home):
        os.makedirs(results_home)

for dir in os.listdir(data_home):
    if "level" in dir and "2023-11-03-18:34:13" not in dir:
        results_dir = os.path.join(results_home, dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        data_dir = os.path.join(data_home, dir)
        for seed in os.listdir(data_dir):
            output_seed_dir = os.path.join(results_dir, seed)
            data_seed_dir = os.path.join(data_dir, seed)
            if not os.path.exists(output_seed_dir):
                os.makedirs(output_seed_dir)
            results_path = os.path.join(output_seed_dir, "results.pkl")
            data_path = os.path.join(data_seed_dir, "results.pkl")
            shutil.copy2(data_path, results_path)
