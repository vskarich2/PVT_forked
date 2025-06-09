import numpy as np
import wandb
from matplotlib import pyplot as plt


class WandbMixin:
    """
    Mixin providing a method to initialize a Weights & Biases run.
    Assumes the inheriting class has `self.checkpoint_folder` and `self.args` attributes.
    """

    def log_saliency(self, epoch):
        saliency_examples = self.collect_saliency_examples()
        saliency_items = self.generate_saliency_from_items(saliency_examples)

        saliency_images = [
            self.plot_three_stage_saliency_wandb(item)
            for item in saliency_items
        ]

        wandb.log({
            "Saliency Examples": saliency_images,
            "epoch": epoch
        })

    def log_per_class_accuracies(self, test_true, test_pred, epoch):

        y_true = np.concatenate(test_true)
        y_pred = np.concatenate(test_pred)

        num_classes = len(self.class_names)
        correct_by_class = np.zeros(num_classes)
        total_by_class = np.zeros(num_classes)

        for yt, yp in zip(y_true, y_pred):
            total_by_class[yt] += 1
            if yt == yp:
                correct_by_class[yt] += 1

        accuracy_by_class = correct_by_class / (total_by_class + 1e-8)
        accuracy_by_class = np.round(accuracy_by_class, 4)

        # Create a DataFrame for just this epoch
        import pandas as pd
        df = pd.DataFrame({
            "class": self.class_names,
            "accuracy": accuracy_by_class,
            "num_samples": total_by_class.astype(int)
        })

        # Sort or not depending on preference
        df = df.sort_values(by="accuracy", ascending=False)

        # Convert to W&B table
        wandb_table = wandb.Table(dataframe=df)

        wandb.log({
            f"Per-Class Accuracy/Epoch {epoch}": wandb.plot_table(
                "wandb/bar/v1",
                wandb_table,
                {"x": "class", "y": "accuracy", "extra": ["num_samples"]}
            ),
            "epoch": epoch
        })

    def log_misclassified(self, mis_examples, epoch):
        # --- Log misclassified examples (up to 5)
        imgs = []
        for pc, t, p in mis_examples:
            colors = pc[:, 2]
            fig = plt.figure(figsize=(4, 4), dpi=120)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, cmap='viridis', s=12, alpha=0.9)
            ax.set_title(f"T:{self.class_names[t]} → P:{self.class_names[p]}", fontsize=10)
            ax.axis("off")
            plt.colorbar(sc, ax=ax, shrink=0.6, label="Height")
            plt.tight_layout()
            imgs.append(wandb.Image(fig))
            plt.close(fig)

        wandb.log({
            f"Misclassifications/{self.args.dataset}": imgs,
            "epoch": epoch
        })

    def log_confusion_matrix(self, test_true, test_pred, epoch):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        y_true = np.concatenate(test_true)
        y_pred = np.concatenate(test_pred)

        # --- Log confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, ax=ax_cm,
            cmap="Blues", vmin=0, vmax=1,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Recall"}
        )
        ax_cm.set_title(f"Epoch {epoch + 1} Confusion Matrix (Norm)")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        wandb.log({
            f"Confusion Matrix/{self.args.dataset}": wandb.Image(fig_cm, caption=f"Epoch {epoch + 1} CM"),
            "epoch": epoch
        })
        plt.close(fig_cm)

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

        # create one Table for per-class stats across all epochs
        self.per_class_table = wandb.Table(columns=["epoch", "class", "accuracy", "num_samples"])
        # tell W&B that anything under “Per-Class Accuracy” uses the ‘epoch’ column as its x-axis
        wandb.define_metric("Per-Class Accuracy/*", step_metric="epoch")




