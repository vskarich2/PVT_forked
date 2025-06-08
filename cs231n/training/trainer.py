from __future__ import print_function

import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import wandb
from matplotlib import pyplot as plt
# ignore everything
from tqdm.auto import tqdm, trange
import torch.nn.functional as F
from cs231n.training.confusion import ConfusionMatrixMixin
from cs231n.training.dataloaders import DataLoaderMixin
from cs231n.training.preprocess_data import DataPreprocessingMixin
from cs231n.training.saliency import SaliencyMixin
from cs231n.training.stats import StatsMixin
from cs231n.training.wandb_logging import WandbMixin

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
import os
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.pvt import pvt
import numpy as np
import sklearn.metrics as metrics
import datetime

# Define ANSI codes:

class Trainer(
    ConfusionMatrixMixin,
    WandbMixin,
    DataLoaderMixin,
    SaliencyMixin,
    DataPreprocessingMixin,
    StatsMixin
):

    def __init__(self, args, io):
        self.args = args
        self.io = io
        self.device = torch.device(self.args.device)
        self.model = self.load_model(self.device)

        if self.args.scanobject_compare:
            print("Registering hooks for saliency gradients!!")
            self.register_saliency_hooks()
            self.register_voxelization_hooks()

        self.opt = self.set_optimizer(self.model)
        self.scheduler = CosineAnnealingLR(self.opt, self.args.epochs, eta_min=self.args.lr)
        self.criterion = self.cal_loss
        self.checkpoint_folder = self.create_checkpoint_folder_name()

        if self.args.wandb:
            self.start_wandb()

    def fit(self):

        print(f"\nTraining Run Starting with....{self.args.dataset}")

        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()

        best_test_acc = 0.0

        # Outer epoch loop with trange
        for epoch in trange(
                self.args.epochs,
                desc="Epoch",
                leave=True,
                unit="epoch"
        ):
            # Train one epoch
            train_avg_loss = self.train_one_epoch(epoch)

            # Test one epoch
            test_acc = self.test_one_epoch(
                epoch,
                train_avg_loss,
                best_test_acc
            )

            # Possibly save new checkpoint
            if float(test_acc) >= float(best_test_acc):
                best_test_acc = float(test_acc)
                self.save_new_checkpoint(epoch, test_acc)

            if self.args.wandb:
                wandb.log({
                    "train/TrainAvgLoss": train_avg_loss,
                    "train/LearningRate": self.scheduler.get_lr()[0],
                    "epoch": epoch
                })
    def stand_alone_test(self):

        print(f"\nPure Testing Run Starting with....{self.args.dataset}")

        test_loader = self.get_test_loader()

        self.model.eval()
        test_true = []
        test_pred = []

        with tqdm(test_loader, unit="batch") as logging_wrapper:
            logging_wrapper.set_description(f"TESTING {len(test_loader)} Batches...")

            with torch.no_grad():
                for data, label in logging_wrapper:
                    (feats, coords), label = self.preprocess_test_data(data, label)

                    feats = feats.to(self.device, non_blocking=True)
                    coords = coords.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)

                    logits = self.model(feats)
                    preds = logits.max(dim=1)[1]
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
                print(outstr)

    def test_one_epoch(
            self,
            epoch,
            train_avg_loss,
            best_test_acc
    ):
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        mis_examples = []  # will store up to 5 mis‐classified: (points, true_lbl, pred_lbl)
        self.model.eval()

        test_bar = tqdm(
            self.test_loader,
            desc=f"Testing  (Epoch {epoch + 1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        with torch.no_grad():
            for data, label in test_bar:
                (feats, coords), label = self.preprocess_test_data(data, label)
                feats = feats.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                logits = self.model(feats)
                loss = self.criterion(logits, label)
                test_loss += loss.item()
                preds = logits.argmax(dim=1)

                count += 1.0
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.cpu().numpy())

                # collect mis‐classifications (up to 5 total)
                if len(mis_examples) < 5:
                    diff = (preds != label).nonzero(as_tuple=False).squeeze(1)
                    for b in diff.tolist():
                        if len(mis_examples) >= 5:
                            break
                        # coords[b]: (3, N_points) → transpose to (N_points,3)
                        pc = coords[b].cpu().numpy().T
                        mis_examples.append((pc, label[b].item(), preds[b].item()))

                running_avg_loss = test_loss / count
                test_bar.set_postfix(test_loss=running_avg_loss)

        test_bar.close()

        # aggregate all predictions / truths
        y_true = np.concatenate(test_true)
        y_pred = np.concatenate(test_pred)

        # 1) Confusion Matrix + W&B logging
        if self.args.conf_matrix:
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

            wandb.log(
                {f"Confusion Matrix/{self.args.dataset}":
                     wandb.Image(fig_cm, caption=f"Epoch {epoch + 1} CM")},
                step=epoch
            )
            plt.close(fig_cm)

            # 2) Mis‐classified examples
            imgs = []
            for pc, t, p in mis_examples:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(3, 3))
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c="gray", s=2)
                ax.set_title(f"T: {self.class_names[t]}\nP: {self.class_names[p]}", fontsize=8)
                ax.axis("off")
                imgs.append(wandb.Image(fig))
                plt.close(fig)

            wandb.log(
                {f"Misclassifications/{self.args.dataset}": imgs},
                step=epoch
            )

        # 3) Compute scalar metrics and return
        test_acc = self.check_stats(
            count,
            epoch,
            test_loss,
            test_pred,
            test_true,
            train_avg_loss,
            best_test_acc
        )

        return test_acc

    def train_one_epoch(self, epoch):

        self.scheduler.step()
        self.model.train()

        # Inner batch loop wrapped in tqdm
        train_bar = tqdm(
            self.train_loader,
            desc=f"Training (Epoch {epoch + 1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        running_loss = 0.0
        running_count = 0.0

        for data, label in train_bar:
            (feats, coords), label = self.preprocess_data(data, label)

            feats = feats.to(self.device, non_blocking=True)
            coords = coords.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            self.opt.zero_grad()
            logits = self.model(feats)
            loss = self.criterion(logits, label)
            loss.backward()
            self.opt.step()

            curr_loss = loss.item()

            running_loss += curr_loss
            running_count += 1.0
            running_avg = running_loss / running_count

            train_bar.set_postfix({
                "Avg Loss": f"🔥{running_avg:.4f}🔥",
                "Batch Loss": f"{curr_loss:.4f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.3e}"
            })

        # Close training bar for this epoch
        train_bar.close()

        # Return final average loss value
        return (running_loss / running_count)

    def create_checkpoint_folder_name(self):
        now = datetime.datetime.now()
        timestamp = now.strftime("%a_%b%d_%Y-%H%M%S")
        drive_location = "/content/drive/My Drive/cs231n_final_project/checkpoints"

        if self.args.use_dsva:
            attn = "dsva"
        else:
            attn = "window"

        dir_name = f'{timestamp}_{attn}_{self.args.dataset}_{self.args.exp_name}'
        save_dir = f'{drive_location}/{dir_name}'
        if not self.args.eval:
            print(f"Saving checkpoints to....{save_dir}")
            print("When you see the ✅ it means the checkpoint was saved!!")

        return save_dir

    def save_new_checkpoint(self, epoch, test_acc):
        model_filename = f"model_epoch_{epoch}_testAcc_{test_acc:.4f}.pth"

        full_checkpoint_path = os.path.join(self.checkpoint_folder, model_filename)
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, full_checkpoint_path)

    def load_model(self, device):
        # Try to load models
        if self.args.model == 'pvt':
            num_classes = 15 if self.args.dataset == 'scanobjectnn' else 40
            model = pvt(args=self.args, num_classes=num_classes).to(device)
        else:
            raise Exception("Not implemented")
        if self.args.use_checkpoint:
            print(f"Loading checkpoint....{self.args.model_path}")
            state_dict = torch.load(self.args.model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"NO CHECKPOINT: Loading fresh model!!")

        return model

    def set_optimizer(self, model):
        if self.args.use_sgd:
            if not self.args.eval:
                print("Using SGD")
            opt = optim.SGD(
                model.parameters(),
                lr=self.args.lr * 10,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        else:
            if not self.args.eval:
                print("Using AdamW")
            opt = optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        return opt

    def cal_loss(self, pred, gold, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''


        gold = gold.to(pred.device)
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss
