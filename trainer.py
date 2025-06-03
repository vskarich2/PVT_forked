from __future__ import print_function
import warnings
# ignore everything
from tqdm.auto import tqdm, trange

warnings.filterwarnings("ignore")
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNetDataLoader
from torch.cuda.amp import autocast, GradScaler
from model.pvt import pvt
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import provider
import datetime
import wandb

# Define ANSI codes:

class Trainer():

    def __init__(self, args, io):
        self.args = args
        self.io = io
        self.device = torch.device(self.args.device)
        self.model = self.load_model(self.device)
        self.opt = self.set_optimizer(self.model)
        self.scheduler = CosineAnnealingLR(self.opt, self.args.epochs, eta_min=self.args.lr)
        self.criterion = cal_loss
        self.checkpoint_folder = self.create_checkpoint_folder_name()

        # ==== AMP CHANGE #2: create a GradScaler
        # only useful if CUDA is available
        if self.device.type == 'cuda' and self.args.amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        # ==== END AMP CHANGE

        if self.args.wandb:
            self.start_wandb()

    def start_wandb(self):
        wandb.init(
            project="cs231n_final_project",
            name=self.checkpoint_folder,
            config={  # optional dictionary of hyperparameters
                "learning_rate": self.args.lr,
                "scheduler": "CosineAnnealingLR",
                "weight_decay": self.args.weight_decay,
                "batch_size": self.args.batch_size,
                "epochs": self.args.epochs
            }
        )

        # This logs weights and gradients every epoch
        wandb.watch(self.model, log="all", log_freq=1)

    def test(self):

        print("Testing Run Starting....")

        test_loader = self.get_test_loader()

        self.model.eval()
        test_true = []
        test_pred = []

        with tqdm(test_loader, unit="batch") as logging_wrapper:
            logging_wrapper.set_description(f"TESTING {len(test_loader)} Batches...")

            with torch.no_grad():
                for data, label in logging_wrapper:

                    label = torch.LongTensor(label[:, 0].numpy())
                    data, label = data.to(self.device), label.to(self.device).squeeze()
                    data = data.permute(0, 2, 1)

                    # ==== AMP CHANGE #3: wrap inference in autocast (optional, but saves memory)
                    if self.device.type == 'cuda' and self.args.amp:
                        with autocast(enabled=(self.device.type == 'cuda')):
                            logits = self.model(data)
                    else:
                        logits = self.model(data)
                    # ==== END AMP CHANGE

                    preds = logits.max(dim=1)[1]
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
                print(outstr)

    def fit(self):

        print("\nTraining Run Starting....")

        print("Setting up dataloaders...")
        train_loader = self.get_train_loader()
        test_loader = self.get_test_loader()

        best_test_acc = 0.0

        # Outer epoch loop with trange
        for epoch in trange(
                self.args.epochs,
                desc="Epoch",
                leave=True,
                unit="epoch"
        ):
            # Train one epoch
            train_avg_loss = self.train_one_epoch(epoch, train_loader)

            # Test one epoch
            test_acc = self.test_one_epoch(
                epoch,
                test_loader,
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

    def test_one_epoch(
            self,
            epoch,
            test_loader,
            train_avg_loss,
            best_test_acc
    ):

        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        self.model.eval()

        # Inner test loop wrapped in tqdm
        test_bar = tqdm(
            test_loader,
            desc=f"Testing  (Epoch {epoch + 1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        with torch.no_grad():
            for data, label in test_bar:
                data, label = self.preprocess_test_data(data, label)

                # ==== AMP CHANGE #4: wrap inference in autocast (optional)
                if self.device.type == 'cuda' and self.args.amp:
                    with autocast(enabled=(self.device.type == 'cuda')):
                        logits = self.model(data)
                else:
                    logits = self.model(data)
                # ==== END AMP CHANGE

                loss = self.criterion(logits, label)
                test_loss += loss.item()

                preds = logits.max(dim=1)[1]

                count += 1.0
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

                # Update test_bar with current average test loss
                running_avg_loss = test_loss / count
                test_bar.set_postfix(test_loss=running_avg_loss)

        # Close test bar for this epoch
        test_bar.close()

        # Compute final metrics for epoch
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

    def train_one_epoch(self, epoch, train_loader):

        self.scheduler.step()
        self.model.train()

        # Inner batch loop wrapped in tqdm
        train_bar = tqdm(
            train_loader,
            desc=f"Training (Epoch {epoch + 1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        running_loss = 0.0
        running_count = 0.0

        for data, label in train_bar:
            data, label = self.preprocess_data(data, label)
            self.opt.zero_grad()
            # ==== AMP CHANGE #5: forward + loss under autocast
            if self.device.type == 'cuda' and self.args.amp:
                with autocast():
                    logits = self.model(data)
                    loss = self.criterion(logits, label)
                # scale the loss, run backward, step optimizer via scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                # CPU or no‐CUDA fallback
                logits = self.model(data)
                loss = self.criterion(logits, label)
                loss.backward()
                self.opt.step()
            # ==== END AMP CHANGE
            curr_loss = loss.item()
            loss.backward()
            self.opt.step()

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

    def preprocess_test_data(self, data, label):
        label = torch.LongTensor(label[:, 0].numpy())
        data, label = data.to(self.device), label.to(self.device).squeeze()
        data = data.permute(0, 2, 1)
        return data, label

    def check_stats(
            self,
            count,
            epoch,
            test_loss,
            test_pred,
            test_true,
            avg_train_loss,
            best_test_acc
    ):
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        if float(test_acc) < float(best_test_acc):
            test_acc_str = f"TestAcc=🔥{test_acc:.4f}🔥"
        else:
            test_acc_str = f"TestAcc=🔥{test_acc:.4f}🔥✅"

        outstr = (
            f"Epoch {epoch + 1:3d}/{self.args.epochs:3d} "
            f"TrainAvgLoss=🔥{avg_train_loss:.4f}🔥 "
            f"TestLoss={(test_loss / count):.4f} "
            f"TestAvgPerClassAcc={avg_per_class_acc:.4f} "
            f"{test_acc_str} "
        )

        print(outstr)

        if self.args.wandb:
            wandb.log({
                "test/TestAcc": test_acc,
                "test/TestLoss": (test_loss / count),
                "test/TestAvgPerClassAcc": avg_per_class_acc,
                "epoch": epoch
            })

        return test_acc

    def preprocess_data(self, data, label):
        data = data.numpy()
        data = provider.random_point_dropout(data)
        data[:, :, 0:3] = provider.random_scale_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = provider.shift_point_cloud(data[:, :, 0:3])
        data = torch.Tensor(data)
        label = torch.LongTensor(label[:, 0].numpy())
        data, label = data.to(self.device), label.to(self.device).squeeze()
        data = data.permute(0, 2, 1)
        return data, label

    def get_test_loader(self):
        test_loader = DataLoader(ModelNetDataLoader(
            partition='test',
            npoint=self.args.num_points,
            args=self.args
        ),
            num_workers=self.args.num_workers,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        return test_loader


    def get_train_loader(self):
        if self.args.dataset == 'modelnet40':
            ds = ModelNetDataLoader(
                npoint=self.args.num_points,
                partition='train',
                uniform=False,
                normal_channel=True,
                cache_size=15000,
                args=self.args
            )
        elif self.args.dataset == 'scanobjectnn':
            ds = ScanObjectNNDataset(
                npoint=self.args.num_points,
                partition='train',
                args=self.args
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def load_model(self, device):
        # Try to load models
        if self.args.model == 'pvt':
            model = pvt(args=self.args).to(device)
        else:
            raise Exception("Not implemented")
        if self.args.use_checkpoint:
            print(f"Loading checkpoint from Google Drive....{self.args.model_path}")
            model.load_state_dict(
                torch.load(
                    self.args.model_path, map_location=device),
                strict=False)
        else:
            print(f"NO CHECKPOINT: Loading fresh model!!")

        return model

    def set_optimizer(self, model):
        if self.args.use_sgd:
            print("Using SGD")
            opt = optim.SGD(
                model.parameters(),
                lr=self.args.lr * 10,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        else:
            print("Using AdamW")
            opt = optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        return opt
