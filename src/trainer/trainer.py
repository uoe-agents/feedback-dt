from typing import Dict

from transformers import Trainer
from transformers import TrainingArguments

from .evaluator import Evaluator
from src.agent import Agent
from src.agent import AgentInput
from src.collator import Collator
from src.dataset.minari_dataset import MinariDataset


class AgentTrainer(Trainer):
    """
    Class for training an agent.

    Args:
        args (Dict): User-supplied arguments.
        agent (Agent): Agent to train.
        collator (Collator): Collator to use for sampling training data.
        dataset (MinariDataset): Dataset to use for training.
    """

    def __init__(
        self, args: Dict, agent: Agent, collator: Collator, dataset: MinariDataset
    ):
        self.user_args = args

        super().__init__(
            model=agent,
            args=TrainingArguments(
                run_name=self.user_args["run_name"],
                output_dir=self.user_args["output"],
                report_to="none"
                if self.user_args["wandb_mode"] == "disabled"
                else "wandb",
                logging_steps=self.user_args["logging_steps"],
                remove_unused_columns=False,
                num_train_epochs=self.user_args["epochs"],
                per_device_train_batch_size=self.user_args["batch_size"],
                learning_rate=self.user_args["lr"],
                weight_decay=1e-4,
                warmup_ratio=0.1,
                optim="adamw_torch",
                max_grad_norm=0.25,
                save_strategy="no",
                seed=self.user_args["model_seed"],
                data_seed=self.user_args["dataset_seed"],
            ),
            train_dataset=dataset,
            data_collator=collator,
        )

        self.create_callbacks()

    def create_callbacks(self):
        """Add the custom evaluation callback to the trainer."""
        self.add_callback(
            Evaluator(
                user_args=self.user_args,
                collator=self.data_collator,
            )
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for a given agent model and input."""
        input = AgentInput(**inputs)
        output = model(input)
        loss = output["loss"]
        return (loss, output) if return_outputs else loss
