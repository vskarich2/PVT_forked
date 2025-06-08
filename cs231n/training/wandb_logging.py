import wandb
import os
os.environ["WANDB_API_KEY"] = "f2b2e46aacf58f9682c3ead0e09f6f4738421290"

class WandbMixin:
    """
    Mixin providing a method to initialize a Weights & Biases run.
    Assumes the inheriting class has `self.checkpoint_folder` and `self.args` attributes.
    """

    def start_wandb(self):
        """
        Initialize a W&B run using the project's hyperparameters.
        """

        wandb.init(
            project="cs231n_final_project",
            name=self.checkpoint_folder,
            config={
                "learning_rate": self.args.lr,
                "scheduler": "CosineAnnealingLR",
                "weight_decay": self.args.weight_decay,
                "batch_size": self.args.batch_size,
                "epochs": self.args.epochs
            }
        )

        # This logs weights and gradients every epoch
        wandb.watch(self.model, log="all", log_freq=1)
