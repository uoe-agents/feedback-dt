import os

import torch
from transformers import DecisionTransformerConfig

from src.agent.fdt import MinigridFDTAgent
from src.collator import Collator
from src.constants import ENV_METADATA_PATH
from src.constants import GLOBAL_SEED
from src.dataset.custom_dataset import CustomDataset
from src.trainer import AgentTrainer
from src.utils.argparsing import get_args
from src.utils.utils import frame_size
from src.utils.utils import log
from src.utils.utils import seed

os.environ["WANDB_DISABLED"] = "true"
os.environ["ENV_METADATA_PATH"] = ENV_METADATA_PATH

seed(GLOBAL_SEED)

args = get_args()

for arg, value in args.items():
    print(f"{arg:}\n {value} \n{'==='*20}")

frame_size = frame_size(args)
print(f"Using frame_size: {frame_size}")

args["wandb_mode"] = "disabled"
args["report_to"] = "none"

early_stopping_patience_factor = (
    4
    if args["eps_per_seed"] == 100
    else (1 if args["eps_per_seed"] == 10 and "GoTo" in args["level"] else 2)
)
args["early_stopping_patience"] = (
    args["early_stopping_patience"] * early_stopping_patience_factor
)

print(f"Using early_stopping_patience: {args['early_stopping_patience']}")

log("setting up devices")
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_str = f"{device.type}:{device.index}" if device.index else f"{device.type}"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    # log(f"Using device: {torch.cuda.get_device_name()}")
    log("Using gpu")
elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    device = torch.device("cpu")
    log("Using cpu")
else:
    device = torch.device("mps")
    log("Using mps")

log("getting dataset...")
dataset = CustomDataset.get_dataset(args)
log("creating collator...")
collator = Collator(
    custom_dataset=dataset,
    args=args,
)

seed(args["model_seed"])

log("creating agent...")
agent = MinigridFDTAgent(
    config=DecisionTransformerConfig(
        state_dim=collator.state_dim,
        act_dim=collator.act_dim,
        state_shape=(3, frame_size, frame_size),
        max_length=args["context_length"],
    ),
    use_missions=args["use_mission"],
    use_feedback=args["use_feedback"],
    use_rtg=args["use_rtg"],
    loss_mean_type=args["loss_mean_type"],
    use_rgb=args["rgb_obs"],
)

log("creating trainer...")
trainer = AgentTrainer(
    agent=agent,
    collator=collator,
    dataset=collator.dataset,
    args=args,
)

log("training agent...")
trainer.train()
