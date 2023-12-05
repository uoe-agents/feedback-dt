from src.dataset.seeds import LEVELS_CONFIGS
from src.dataset.seeds import SeedFinder

if __name__ == "__main__":
    for level in LEVELS_CONFIGS["original_tasks"]:
        print(level)
        seed_finder = SeedFinder(n_train_seeds_required=1280)
        seed_finder.find_seeds(level=level)
