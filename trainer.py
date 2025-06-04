from __future__ import print_function
import warnings

# ignore everything
#from tqdm.auto import tqdm, trange

from tqdm import tqdm, trange
import torch

from data import ScanObjectNNDatasetModified

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
import os
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNetDataLoader, ScanObjectNNDataset
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

        # scheduler: pure CosineAnnealingLR over all epochs, eta_min = self.args.lr
        self.scheduler = CosineAnnealingLR(self.opt, T_max=self.args.epochs, eta_min=self.args.lr)

        self.criterion = cal_loss
        self.checkpoint_folder = self.create_checkpoint_folder_name()

        # ==== AMP CHANGE #2: create a GradScaler only if CUDA+AMP
        if self.device.type == 'cuda' and self.args.amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        # ==== END AMP CHANGE

        if self.args.wandb:
            self.start_wandb()

    def choose_scheduler(self):


        sched = CosineAnnealingLR(
            self.opt,
            self.args.epochs,
            eta_min=self.args.lr
        )
        return sched

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

    def get_train_loader(self):
        """
        - ModelNet40: use ModelNetDataLoader
        - ScanObjectNN: use ScanObjectNNDataset (supports optional dev subset and knn_normals)
        """
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
            # If the user set dev_scan_subset, load the dev subset path; otherwise load full ScanObjectNN
            if getattr(self.args, 'dev_scan_subset', False):
                base = os.path.dirname(os.path.abspath(__file__))
                subset_root = os.path.join(base, 'data', 'dev_scanObjectNN_subset')
                print("USING DEV SCAN SUBSET at", subset_root)
                ds = ScanObjectNNDataset(
                    npoint=self.args.num_points,
                    partition='train',
                    args=self.args,
                    knn_normals=self.args.knn_normals if hasattr(self.args, 'knn_normals') else 30
                )
            else:
                ds = ScanObjectNNDataset(
                    npoint=self.args.num_points,
                    partition='train',
                    args=self.args,
                    knn_normals=self.args.knn_normals if hasattr(self.args, 'knn_normals') else 30
                )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.args.persist_workers,
            prefetch_factor=self.args.prefetch_factor,
            num_workers=self.args.num_workers
        )

    def get_test_loader(self):
        """
        - ModelNet40: use ModelNetDataLoader (partition='test')
        - ScanObjectNN: use ScanObjectNNDataset (partition='test', optional dev subset)
        """
        if self.args.dataset == 'modelnet40':
            ds = ModelNetDataLoader(
                partition='test',
                npoint=self.args.num_points,
                args=self.args
            )
        elif self.args.dataset == 'scanobjectnn' and self.args.scanobject_compare:
            ds = ScanObjectNNDatasetModified(
                npoint=self.args.num_points,
                partition='test',
                args=self.args,
                knn_normals=self.args.knn_normals if hasattr(self.args, 'knn_normals') else 30
            )
        elif self.args.dataset == 'scanobjectnn':
            if getattr(self.args, 'dev_scan_subset', False):
                base = os.path.dirname(os.path.abspath(__file__))
                subset_root = os.path.join(base, 'data', 'dev_scanObjectNN_subset')
                print("USING DEV SCAN SUBSET at", subset_root)
                ds = ScanObjectNNDataset(
                    npoint=self.args.num_points,
                    partition='test',
                    args=self.args,
                    knn_normals=self.args.knn_normals if hasattr(self.args, 'knn_normals') else 30
                )
            else:
                ds = ScanObjectNNDataset(
                    npoint=self.args.num_points,
                    partition='test',
                    args=self.args,
                    knn_normals=self.args.knn_normals if hasattr(self.args, 'knn_normals') else 30
                )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        return DataLoader(
            ds,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.args.persist_workers,
            prefetch_factor=self.args.prefetch_factor
        )

    def preprocess_data(self, data, label):
        """
        For ModelNet40: `data` comes in as (B, N, 6) → we split into
            feats  = (B, 6, N)   and  coords = (B, 3, N).
        For ScanObjectNN: `data` comes in as (B, N, 6) → we split into
            feats  = (B, 6, N)   and  coords = (B, 3, N).
        """
        # 1) NumPy-side augmentations
        data_np = data.numpy()                               # (B, N, C)
        data_np = provider.random_point_dropout(data_np)
        data_np[:, :, 0:3] = provider.random_scale_point_cloud(data_np[:, :, 0:3])
        data_np[:, :, 0:3] = provider.shift_point_cloud(data_np[:, :, 0:3])

        # 2) back to FloatTensor
        data_t = torch.from_numpy(data_np.astype('float32'))  # (B, N, C)

        # 3) split into feats / coords depending on dataset
        if self.args.dataset == 'modelnet40':
            # feats = all 6 channels, transposed to (B, 6, N)
            feats = data_t.permute(0, 2, 1).to(self.device)           # (B, 6, N)
            # coords = first-three dims, transposed to (B, 3, N)
            coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)
        elif self.args.dataset == 'scanobjectnn':
            # ScanObjectNNDataset already returns (npoint, 6), so data_t is (B, N, 6)
            # feats = all 6 channels, transposed to (B, 6, N)
            feats = data_t.permute(0, 2, 1).to(self.device)  # (B, 6, N)
            # coords = first-three dims, transposed to (B, 3, N)
            coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        # 4) turn `label` into a 1D LongTensor of shape (B,)
        label_tensor = label.long().to(self.device)

        return (feats, coords), label_tensor

    def preprocess_test_data(self, data, label):
        """
        Exactly the same splitting logic as preprocess_data, but no random augmentations.
        """
        data_np = data.numpy()                                # (B, N, C)
        data_t = torch.from_numpy(data_np.astype('float32'))   # (B, N, C)

        if self.args.dataset == 'modelnet40':
            feats = data_t.permute(0, 2, 1).to(self.device)           # (B, 6, N)
            coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)
        elif self.args.dataset == 'scanobjectnn':
            feats = data_t.permute(0, 2, 1).to(self.device)  # (B, 6, N)
            coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        label_tensor = torch.LongTensor(label).to(self.device)  # (B,)
        return (feats, coords), label_tensor

    def train_one_epoch(self, epoch, train_loader):
        self.scheduler.step()
        self.model.train()

        train_bar = tqdm(
            train_loader,
            desc=f"Training (Epoch {epoch + 1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        running_loss = 0.0
        running_count = 0.0

        
        for data, label in train_bar:
            (feats, coords), label = self.preprocess_data(data, label)
            self.opt.zero_grad()
            # ==== AMP CHANGE #5: forward + loss under autocast
            if self.device.type == 'cuda' and self.args.amp:
                with autocast():
                    logits = self.model(feats)
                    loss = self.criterion(logits, label)
                # scale the loss, run backward, step optimizer via scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                # CPU or no‐CUDA fallback
                logits = self.model(feats)
                loss = self.criterion(logits, label)
                loss.backward()
                self.opt.step()
            # ==== END AMP CHANGE
        

            curr_loss = loss.item()
            running_loss += curr_loss
            running_count += 1.0
            running_avg = running_loss / running_count

            train_bar.set_postfix({
                "Avg Loss": f"🔥{running_avg:.4f}🔥",
                "Batch Loss": f"{curr_loss:.4f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.3e}"
            })

        train_bar.close()
        return (running_loss / running_count)

    def test_one_epoch(self, epoch, test_loader, train_avg_loss, best_test_acc):
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        self.model.eval()

        test_bar = tqdm(
            test_loader,
            desc=f"Testing  (Epoch {epoch + 1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        with torch.no_grad():
            for data, label in test_bar:
                (feats, coords), label = self.preprocess_test_data(data, label) 

                # ==== AMP CHANGE #4: wrap inference in autocast (optional)
                if self.device.type == 'cuda' and self.args.amp:
                    with autocast():
                        logits = self.model(feats)
                else:
                    logits = self.model(feats)
                # ==== END AMP CHANGE

                loss = self.criterion(logits, label)
                test_loss += loss.item()

                preds = logits.max(dim=1)[1]
                count += 1.0
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

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

    def test_one_epoch_compare(self, epoch, test_loader, train_avg_loss, best_test_acc):
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []
        self.model.eval()

        test_bar = tqdm(
            test_loader,
            desc=f"Testing  (Epoch {epoch + 1}/{self.args.epochs})",
            leave=False,
            unit="batch"
        )
        with torch.no_grad():
            for data, label in test_bar:
                (feats, coords), label = self.preprocess_test_data(data, label)

                # ==== AMP CHANGE #4: wrap inference in autocast (optional)
                if self.device.type == 'cuda' and self.args.amp:
                    with autocast():
                        logits = self.model(feats)
                else:
                    logits = self.model(feats)
                # ==== END AMP CHANGE

                loss = self.criterion(logits, label)
                test_loss += loss.item()

                preds = logits.max(dim=1)[1]
                count += 1.0
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

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

        if self.args.wandb:
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
            if self.args.wandb:
                wandb.log({
                    "train/TrainAvgLoss": train_avg_loss,
                    "train/LearningRate": current_lr,
                    "epoch": epoch
                })

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
                    (feats, coords), label = self.preprocess_test_data(data, label)
                    logits = self.model(feats) 
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

    def test_compare_with_hooks(self):
        """
        Run through the ScanObjectNN test set, and for each example, do:
          - Forward pass → get `logits`.
          - Pick the predicted‐class logit, call `.backward()`.
          - Read out `model.grad_enc1`, `model.grad_enc2`, `model.grad_enc3`,
            which now contain gradient tensors of shape [1, Ck, R, R, R].
          - Mask out “empty” voxels and convert each [Ck, R³] → [V_occ, Ck].
          - Return a list of dicts, one per test‐example, containing:
                { "pred": int,
                  "true": int,
                  "classname": str,
                  "coords": (V_occ, 3),
                  "feat_enc1": (V_occ, C1),  "grad_enc1": (V_occ, C1),
                  "feat_enc2": (V_occ, C2),  "grad_enc2": (V_occ, C2),
                  "feat_enc3": (V_occ, C3),  "grad_enc3": (V_occ, C3)
                }
        """

        self.model.eval()
        test_loader = self.get_test_loader()

        all_results = []
        total_true = []
        total_pred = []

        for (data, label, classname) in tqdm(test_loader, desc="Testing Batches"):
            # 1) Preprocess raw points → (feats, coords) exactly as training does
            (feats, coords), label = self.preprocess_test_data(data, label)
            # feats: [B, C_point, N_pts], coords: [B, N_pts, 3], label: [B]

            feats = feats.to(self.device)
            coords = coords.to(self.device)
            label = label.to(self.device)

            B, _, _ = feats.shape

            # a) Zero out any previously stored gradients/hooks
            self.model.grad_enc1 = None
            self.model.grad_enc2 = None
            self.model.grad_enc3 = None
            self.model.zero_grad()

            # b) Forward pass through the entire network
            logits = self.model(feats, coords)  # shape [B, num_classes]
            preds = logits.argmax(dim=1)  # shape [B]

            total_true.append(label.cpu().numpy())
            total_pred.append(preds.cpu().numpy())

            # c) For each sample in the batch, do a backward on that sample’s logit
            for i in range(B):
                # Zero‐grad activations from previous iteration
                self.model.zero_grad()
                self.model.grad_enc1 = None
                self.model.grad_enc2 = None
                self.model.grad_enc3 = None

                # Pick out the i‐th logit for the predicted class
                pred_class_i = preds[i].item()
                scalar_logit = logits[i, pred_class_i]  # a scalar tensor

                # Backward this scalar → populates grad_enc1/2/3
                scalar_logit.backward(retain_graph=True)

                # By now, after backward, each model.grad_encK is a [B, Ck, R, R, R] tensor
                g1 = self.model.grad_enc1[i].detach().cpu()  # [C1, R, R, R]
                g2 = self.model.grad_enc2[i].detach().cpu()  # [C2, R, R, R]
                g3 = self.model.grad_enc3[i].detach().cpu()  # [C3, R, R, R]

                # Also pull the forward activation (we laced them into self._actK)
                a1 = self.model._act1[i].detach().cpu()  # [C1, R, R, R]
                a2 = self.model._act2[i].detach().cpu()  # [C2, R, R, R]
                a3 = self.model._act3[i].detach().cpu()  # [C3, R, R, R]

                #  d) Flatten each [Ck, R, R, R] → [Ck, R³], then transpose → [R³, Ck]
                C1, Rx, Ry, Rz = a1.shape
                R_cubed = Rx * Ry * Rz

                flat_a1 = a1.reshape(C1, R_cubed).transpose(0, 1)  # [R³, C1]
                flat_g1 = g1.reshape(C1, R_cubed).transpose(0, 1)  # [R³, C1]

                C2, _, _, _ = a2.shape
                flat_a2 = a2.reshape(C2, R_cubed).transpose(0, 1)  # [R³, C2]
                flat_g2 = g2.reshape(C2, R_cubed).transpose(0, 1)  # [R³, C2]

                C3, _, _, _ = a3.shape
                flat_a3 = a3.reshape(C3, R_cubed).transpose(0, 1)  # [R³, C3]
                flat_g3 = g3.reshape(C3, R_cubed).transpose(0, 1)  # [R³, C3]

                # e) Build the same “non_empty_mask” that the model uses for voxelization.
                #    Assume the model has a utility for that. For example:
                non_empty_mask = self.model.compute_non_empty_mask(feats[i:i + 1], voxel_coords=vcoords[i:i + 1])
                #    → shape [1, R³], boolean
                mask1d = non_empty_mask.view(-1)  # [R³] on CPU

                # f) Select only the occupied voxels (mask1d == True).  Let V_occ = mask1d.sum().
                occ_idx = torch.nonzero(mask1d, as_tuple=False).squeeze(1).numpy()  # indices of occupied voxels

                coords_np = vcoords[i].detach().cpu().numpy()  # [R³, 3]
                coords_occ = coords_np[occ_idx]  # [V_occ, 3]

                feat1_occ = flat_a1[occ_idx]  # [V_occ, C1]
                grad1_occ = flat_g1[occ_idx]  # [V_occ, C1]

                feat2_occ = flat_a2[occ_idx]  # [V_occ, C2]
                grad2_occ = flat_g2[occ_idx]  # [V_occ, C2]

                feat3_occ = flat_a3[occ_idx]  # [V_occ, C3]
                grad3_occ = flat_g3[occ_idx]  # [V_occ, C3]

                # g) Build a dictionary for this single example
                item = {
                    "pred": pred_class_i,
                    "true": label[i].item(),
                    "classname": classname[i],
                    "coords": coords_occ,  # shape (V_occ, 3)
                    "feat1": feat1_occ,  # shape (V_occ, C1)
                    "grad1": grad1_occ,  # shape (V_occ, C1)
                    "feat2": feat2_occ,  # shape (V_occ, C2)
                    "grad2": grad2_occ,  # shape (V_occ, C2)
                    "feat3": feat3_occ,  # shape (V_occ, C3)
                    "grad3": grad3_occ  # shape (V_occ, C3)
                }
                all_results.append(item)

                # h) Clear out gradients before next loop iteration
                self.model.zero_grad()
                self.model.grad_enc1 = None
                self.model.grad_enc2 = None
                self.model.grad_enc3 = None

        # 13) Compute final accuracy over entire test set
        total_true = np.concatenate(total_true)
        total_pred = np.concatenate(total_pred)
        test_acc = metrics.accuracy_score(total_true, total_pred)
        avg_class = metrics.balanced_accuracy_score(total_true, total_pred)
        print(f"Test :: acc={test_acc:.4f}, avg‐per‐class={avg_class:.4f}")

        return all_results



