# trainer.py

from __future__ import print_function
from tqdm.notebook import tqdm, trange
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
import os
import argparse
import torch
import torch.optim as optim

# ─── Additional imports for schedulers ────────────────────────────────────────
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
# ──────────────────────────────────────────────────────────────────────────────

from data import ModelNetDataLoader, ScanObjectNNDataset
from model.pvt import pvt
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import provider
import datetime
import wandb


class Trainer():
    def __init__(self, args, io):
        self.args = args
        self.io = io
        self.device = torch.device(self.args.device)
        self.model = self.load_model(self.device)

        # 1) Create optimizer (SGD or AdamW)
        self.opt = self.set_optimizer(self.model)

        # 2) Set up scheduler:
        #    - If using SGD → pure CosineAnnealingLR over all epochs (eta_min=0)
        #    - If using AdamW → 10‐epoch linear warmup, then cosine decay
        if self.args.use_sgd:
            # pure cosine decay for SGD: T_max = total epochs, eta_min = 0
            self.scheduler = CosineAnnealingLR(self.opt, T_max=self.args.epochs, eta_min=0.0)
        else:
            # LambdaLR for AdamW: 10‐epoch linear warmup (0→1), then cosine decay to 0
            total_epochs = self.args.epochs
            warmup_epochs = 10

            def lr_lambda(current_epoch):
                if current_epoch < warmup_epochs:
                    # linear warmup: 0 → 1 over first warmup_epochs
                    return float(current_epoch) / float(max(1, warmup_epochs))
                # cosine decay after warmup
                cosine_epochs = total_epochs - warmup_epochs
                epoch_in_cosine = current_epoch - warmup_epochs
                return 0.5 * (1.0 + math.cos(math.pi * epoch_in_cosine / float(cosine_epochs)))

            self.scheduler = LambdaLR(self.opt, lr_lambda=lr_lambda)

        self.criterion = cal_loss
        self.checkpoint_folder = self.create_checkpoint_folder_name()
        self.start_wandb()

    def start_wandb(self):
        sched_name = "Cosine(SGD)" if self.args.use_sgd else "Warmup+Cosine(AdamW)"
        wandb.init(
            project="cs231n_final_project",
            name=self.checkpoint_folder,
            config={
                "learning_rate": self.args.lr,
                "scheduler": sched_name,
                "weight_decay": self.args.weight_decay,
                "batch_size": self.args.batch_size,
                "epochs": self.args.epochs
            }
        )
        wandb.watch(self.model, log="all", log_freq=1)

    def load_model(self, device):
        if self.args.model == 'pvt':
            model = pvt(args=self.args).to(device)
        else:
            raise Exception("Not implemented")
        if self.args.use_checkpoint:
            print(f"Loading checkpoint from Google Drive....{self.args.model_path}")
            model.load_state_dict(
                torch.load(self.args.model_path, map_location=device),
                strict=False
            )
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
            if getattr(self.args, 'dev_scan_subset', False):
                # Use the small .npy “dev” subset
                base = os.path.dirname(os.path.abspath(__file__))
                subset_root = os.path.join(base, 'data', 'dev_scanObjectNN_subset')
                print("USING DEV SCAN SUBSET at", subset_root)
                ds = ScanObjectNNSubset(
                    root_dir=subset_root,
                    npoint=self.args.num_points,
                    partition='train'
                )
            else:
                # The normal, full‐H5 loader
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
            num_workers=self.args.num_workers,
            drop_last=True
        )
        

    def get_test_loader(self):
        if self.args.dataset == 'modelnet40':
            ds = ModelNetDataLoader(
                partition='test',
                npoint=self.args.num_points,
                args=self.args
            )
        elif self.args.dataset == 'scanobjectnn':
            if getattr(self.args, 'dev_scan_subset', False):
                base = os.path.dirname(os.path.abspath(__file__))
                subset_root = os.path.join(base, 'data', 'dev_scanObjectNN_subset')
                print("USING DEV SCAN SUBSET at", subset_root)
                ds = ScanObjectNNSubset(
                    root_dir=subset_root,
                    npoint=self.args.num_points,
                    partition='test'
                )
            else:
                ds = ScanObjectNNDataset(
                    npoint=self.args.num_points,
                    partition='test',
                    args=self.args
                )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        return DataLoader(
            ds,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False
        )

        def preprocess_data(self, data, label):
            """
            For ModelNet40: `data` comes in as (B, N, 6) → we split into
                feats  = (B, 6, N)   and  coords = (B, 3, N).
            For ScanObjectNN: `data` comes in as (B, N, 3) → we set
                feats  = coords = (B, 3, N).
    
            Returns:
                (feats, coords), label_tensor
            """
            # 1) NumPy‐side augmentations
            data_np = data.numpy()                               # (B, N, C)
            data_np = provider.random_point_dropout(data_np)
            data_np[:, :, 0:3] = provider.random_scale_point_cloud(data_np[:, :, 0:3])
            data_np[:, :, 0:3] = provider.shift_point_cloud(data_np[:, :, 0:3])
    
            # 2) back to FloatTensor
            data_t = torch.from_numpy(data_np.astype('float32'))  # (B, N, C)
    
            # 3) split into feats / coords depending on dataset
            if self.args.dataset == 'modelnet40':
                # feats = all 6 channels, transposed to (B, 6, N)
                feats  = data_t.permute(0, 2, 1).to(self.device)           # (B, 6, N)
                # coords = first‐three dims, transposed to (B, 3, N)
                coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)
            elif self.args.dataset == 'scanobjectnn':
                # everything is XYZ, so (B, N, 3) → (B, 3, N)
                coords = data_t.permute(0, 2, 1).to(self.device)  # (B, 3, N)
                feats  = coords.clone()                           # (B, 3, N)
            else:
                raise ValueError(f"Unsupported dataset: {self.args.dataset}")
    
            # 4) turn `label` into a 1D LongTensor of shape (B,)
            label_tensor = torch.LongTensor(label).to(self.device)
    
            return (feats, coords), label_tensor


    def preprocess_test_data(self, data, label):
        """
        Exactly the same splitting logic as preprocess_data, but no random augmentations.
        """
        data_np = data.numpy()                                # (B, N, C)
        data_t  = torch.from_numpy(data_np.astype('float32')) # (B, N, C)

        if self.args.dataset == 'modelnet40':
            feats  = data_t.permute(0, 2, 1).to(self.device)           # (B, 6, N)
            coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)
        elif self.args.dataset == 'scanobjectnn':
            coords = data_t.permute(0, 2, 1).to(self.device)  # (B, 3, N)
            feats  = coords.clone()                           # (B, 3, N)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        label_tensor = torch.LongTensor(label).to(self.device)  # (B,)
        return (feats, coords), label_tensor


    def train_one_epoch(self, epoch, train_loader):
        self.scheduler.step()
        self.model.train()

        train_bar = tqdm(
            train_loader,
            desc=f"Training (Epoch {epoch+1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        running_loss = 0.0
        running_count = 0.0

        for data, label in train_bar:
            (feats, coords), label_tensor = self.preprocess_data(data, label)
            self.opt.zero_grad()

            # pass both feats and coords into the model
            logits = self.model(feats, coords)
            loss = self.criterion(logits, label_tensor)
            loss.backward()
            self.opt.step()

            curr_loss = loss.item()
            running_loss += curr_loss
            running_count += 1.0
            running_avg = running_loss / running_count

            train_bar.set_postfix({
                "Avg Loss": f"{running_avg:.4f}",
                "Batch Loss": f"{curr_loss:.4f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.3e}"
            })

        train_bar.close()
        return running_loss / running_count


    def test_one_epoch(self, epoch, test_loader, train_avg_loss, best_test_acc):
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        self.model.eval()

        test_bar = tqdm(
            test_loader,
            desc=f"Testing  (Epoch {epoch+1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        with torch.no_grad():
            for data, label in test_bar:
                (feats, coords), label_tensor = self.preprocess_test_data(data, label)

                # pass both feats and coords into the model
                logits = self.model(feats, coords)
                loss = self.criterion(logits, label_tensor)
                test_loss += loss.item()

                preds = logits.max(dim=1)[1]
                test_true.append(label_tensor.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                count += 1.0

                running_avg_loss = test_loss / count
                test_bar.set_postfix(test_loss=running_avg_loss)

        test_bar.close()
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


    def check_stats(self, count, epoch, test_loss, test_pred, test_true, avg_train_loss, best_test_acc):
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

        wandb.log({
            "test/TestAcc": test_acc,
            "test/TestLoss": (test_loss / count),
            "test/TestAvgPerClassAcc": avg_per_class_acc,
            "epoch": epoch
        })
        return test_acc

    def fit(self):
        print("\nTraining Run Starting....")
        print("Setting up dataloaders...")
        train_loader = self.get_train_loader()
        test_loader = self.get_test_loader()

        best_test_acc = 0.0

        for epoch in trange(
                self.args.epochs,
                desc="Epoch",
                leave=True,
                unit="epoch"
        ):
            # Step the scheduler at the start of each epoch
            self.scheduler.step()

            train_avg_loss = self.train_one_epoch(epoch, train_loader)

            test_acc = self.test_one_epoch(
                epoch,
                test_loader,
                train_avg_loss,
                best_test_acc
            )

            if float(test_acc) >= float(best_test_acc):
                best_test_acc = float(test_acc)
                self.save_new_checkpoint(epoch, test_acc)

            current_lr = self.scheduler.get_last_lr()[0]
            wandb.log({
                "train/TrainAvgLoss": train_avg_loss,
                "train/LearningRate": current_lr,
                "epoch": epoch
            })

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
                    # Handle both [batch,1] and [batch] here as well
                    if label.dim() == 2 and label.size(1) == 1:
                        label = label[:, 0]
                    label = label.long()

                    data, label = data.to(self.device), label.to(self.device)
                    data = data.permute(0, 2, 1)
                    logits = self.model(data)

                    preds = logits.max(dim=1)[1]
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (
                    test_acc, avg_per_class_acc)
                print(outstr)
