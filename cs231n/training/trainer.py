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
    StatsMixin
):

    def __init__(self, args, io):
        self.args = args
        self.io = io
        self.device = torch.device(self.args.device)
        self.model = self.load_model(self.device)
        print(f"Using device: {self.args.device}")
        self.class_names_modelnet = [
            "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
            "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
            "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
            "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
            "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
            "wardrobe", "xbox"
        ]
        self.class_names_scanobject = [
                                "bag",
                                "bin",
                                "box",
                                "cabinet",
                                "chair",
                                "desk",
                                "display",
                                "door",
                                "shelf",
                                "table",
                                "bed",
                                "pillow",
                                "sink",
                                "sofa",
                                "toilet"
                             ]
        self.class_names = self.class_names_modelnet if self.args.dataset == "modelnet40" else self.class_names_scanobject

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

    def test_one_epoch(self, epoch, train_avg_loss, best_test_acc):
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        mis_examples = []
        correct_items, incorrect_items = [], []
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

            saliency_examples = self.collect_saliency_examples()


            running_avg_loss = test_loss / count
            test_bar.set_postfix(test_loss=running_avg_loss)

        test_bar.close()

        # ───────────────────────────────────────────────────────────────
        # Confusion matrix and W&B logging
        # ───────────────────────────────────────────────────────────────
        if self.args.conf_matrix:
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

    def save_misclassified(self, coords, label, mis_examples, preds):
        # save misclassified examples
        if len(mis_examples) < 5:
            diff = (preds != label).nonzero(as_tuple=False).squeeze(1)
            for b in diff.tolist():
                if len(mis_examples) >= 5:
                    break
                pc = coords[b].cpu().numpy().T
                mis_examples.append((pc, label[b].item(), preds[b].item()))

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
