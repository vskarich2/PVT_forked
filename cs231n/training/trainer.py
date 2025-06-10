from __future__ import print_function

import math
import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import wandb
from matplotlib import pyplot as plt
# ignore everything
from tqdm.auto import tqdm, trange
import torch.nn.functional as F

from cs231n.training.checkpoint_mixin import CheckpointMixin
from cs231n.training.confusion import ConfusionMatrixMixin
from cs231n.training.dataloaders import DataLoaderMixin
from cs231n.training.preprocess_data import DataPreprocessingMixin
from cs231n.training.saliency import SaliencyMixin
from cs231n.training.stand_alone_test_mixin import StandAloneTestMixin
from cs231n.training.stats import StatsMixin
from cs231n.training.wandb_logging import WandbMixin

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
import os
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
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
    StatsMixin,
    CheckpointMixin,
    StandAloneTestMixin
):

    def __init__(self, args, io):
        self.args = args
        self.io = io
        self.device = torch.device(self.args.device)
        self.model = self.load_model(self.device)
        print(f"Using device: {self.args.device}")

        self.class_names = self.get_class_names()

        if self.args.compute_saliency:
            print("Registering hooks for saliency gradients!!")
            self.register_saliency_hooks()
            self.register_voxelization_hooks()

        self.opt, self.scheduler = self.get_optimizer_and_scheduler(self.model)
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
                lr = self.scheduler.get_last_lr()[0]
                wandb.log({
                    "train/TrainAvgLoss": train_avg_loss,
                    "train/LearningRate": f"{lr:.4f}",
                    "epoch": epoch
                })

    def test_one_epoch(self, epoch, train_avg_loss, best_test_acc):
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        mis_examples = []
        self.model.eval()

        test_bar = tqdm(
            self.test_loader,
            desc=f"Testing  (Epoch {epoch + 1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        with torch.no_grad():
            for data, label, *maybe_class_name in test_bar:
                class_name = maybe_class_name[0] if maybe_class_name else "NONE"

                (feats, coords), label = self.preprocess_test_data(data, label)
                feats = feats.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                logits = self.model(feats)
                loss = self.criterion(logits, label)
                test_loss += loss.item()
                preds = logits.argmax(dim=1)
                self.save_misclassified(coords, label, mis_examples, preds)

                count += 1.0
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.cpu().numpy())
                running_avg_loss = test_loss / count
                test_bar.set_postfix(test_loss=running_avg_loss)

            if self.args.compute_saliency and self.args.wandb:
                self.log_saliency(epoch)


        test_bar.close()

        # ───────────────────────────────────────────────────────────────
        # Confusion matrix and W&B logging
        # ───────────────────────────────────────────────────────────────
        if self.args.conf_matrix and self.args.wandb:
            self.log_confusion_matrix(test_true, test_pred, epoch)
            self.log_misclassified(mis_examples, epoch)
            self.log_per_class_accuracies(test_true, test_pred, epoch)


        # Final scalar metrics
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

            lr = self.scheduler.get_last_lr()[0]
            train_bar.set_postfix({
                "Avg Loss": f"🔥{running_avg:.4f}🔥",
                "Batch Loss": f"{curr_loss:.4f}",
                "LR": f"{lr:.4f}"
            })

        # Log gradient and param stats
        #if self.args.wandb:
            #self.log_gradient_and_param_statistics(epoch=epoch)

        # Close training bar for this epoch
        train_bar.close()

        # Return final average loss value
        return (running_loss / running_count)

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

    def get_optimizer_and_scheduler(self, model):
        if self.args.use_sgd:
            if not self.args.eval:
                print("Using SGD")

            optimizer = optim.SGD(
                model.parameters(),
                lr=self.args.lr * 10,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
            scheduler = CosineAnnealingLR(optimizer, self.args.epochs, eta_min=self.args.lr)
        else:
            if not self.args.eval:
                print("Using AdamW with 10 epoch warm up followed by CosineAnnealingLR.")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )

            warmup_epochs = 10
            total_epochs = 200

            # 1) Warmup from 0 → 1×LR over first warmup_epochs
            warmup_sched = LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_epochs
            )

            # 2) Cosine anneal from 1×LR → η_min over the rest
            cosine_sched = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=1e-5
            )

            # 3) Chain them together
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs]
            )

        return optimizer, scheduler

    def cal_loss(self, pred, gold, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''


        gold = gold.to(pred.device)
        gold = gold.contiguous().reshape(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.reshape(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss
