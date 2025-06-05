from __future__ import print_function

import warnings

import torch
from matplotlib import pyplot as plt
# ignore everything
from tqdm.auto import tqdm, trange

from PVT_forked.modules.dsva.confusion_matrix import plot_confusion_matrix
from PVT_forked.modules.dsva.dsva_cross_attention import SparseDynamicVoxelAttention

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
import os
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNetDataset, ScanObjectNNDataset, ScanObjectNNDatasetModified
from torch.cuda.amp import autocast, GradScaler
from model.pvt import pvt
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss
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

        if self.args.scanobject_compare:
            self.register_saliency_hooks()
            self.register_voxelization_hooks()

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

    def compute_confusion_matrix(model, dataloader, device, num_classes):
        """
        Run inference on the given dataloader and compute the confusion matrix.

        Args:
            model (torch.nn.Module): trained classification model.
            dataloader (torch.utils.data.DataLoader): DataLoader for ModelNet40 test set.
            device (torch.device): torch device (e.g., torch.device('cuda')).
            num_classes (int): number of classes (40 for ModelNet40).
        Returns:
            cm (np.ndarray): confusion matrix shape (num_classes, num_classes),
                             where cm[i, j] = count of true label i predicted as j.
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for points, labels in dataloader:
                # points: [B, N, 3] or however your dataset returns them
                # labels: [B]
                points = points.to(device)  # e.g., [B, 1024, 3]
                labels = labels.to(device)  # [B]

                # Forward pass
                logits = model(points)  # assume output shape [B, num_classes]
                preds = logits.argmax(dim=1)  # [B]

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)  # shape [num_samples]
        all_labels = np.concatenate(all_labels, axis=0)  # shape [num_samples]

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        return cm

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
                    # ==== AMP CHANGE #3: wrap inference in autocast (optional, but saves memory)
                    if self.device.type == 'cuda' and self.args.amp:
                        with autocast(enabled=(self.device.type == 'cuda')):
                            logits = self.model(feats)
                    else:
                        logits = self.model(feats)
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

    def make_confusion_matrix_for_modelnet(self):

        print("Generating Confusion Matrix for ModelNet40....")

        test_loader = self.get_test_loader()
        self.model.eval()

        all_preds = []
        all_labels = []
        test_true = []
        test_pred = []

        # Standard ModelNet40 class names in label order
        class_names = [
            "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
            "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
            "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
            "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
            "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
            "wardrobe", "xbox"
        ]

        # Will collect up to 5 wrongly classified examples: (true_name, predicted_name)
        wrong_examples = []

        with tqdm(test_loader, unit="batch") as logging_wrapper:
            logging_wrapper.set_description(f"TESTING {len(test_loader)} Batches...")

            with torch.no_grad():
                for data, label in logging_wrapper:
                    # Preprocess exactly as training
                    (feats, coords), label = self.preprocess_test_data(data, label)

                    feats = feats.to(self.device)
                    label = label.to(self.device)

                    # Forward pass
                    logits = self.model(feats)
                    preds = logits.argmax(dim=1)

                    # Accumulate for confusion matrix
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(label.cpu().numpy())

                    # Check up to 5 misclassifications
                    if len(wrong_examples) < 5:
                        preds_cpu = preds.detach().cpu()
                        label_cpu = label.cpu()
                        for i in range(label_cpu.size(0)):
                            if len(wrong_examples) >= 5:
                                break
                            true_idx = label_cpu[i].item()
                            pred_idx = preds_cpu[i].item()
                            if pred_idx != true_idx:
                                true_name = class_names[true_idx]
                                pred_name = class_names[pred_idx]
                                wrong_examples.append((true_name, pred_name))

                # Concatenate after all batches
                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                # Compute and print metrics
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = f"Test :: test acc: {test_acc:.6f}, test avg acc: {avg_per_class_acc:.6f}"
                print(outstr)

                # Plot confusion matrix
                cm = confusion_matrix(all_labels, all_preds, labels=list(range(40)))
                plot_confusion_matrix(cm, class_names)
                plt.savefig("modelnet40_confusion_matrix.png", dpi=300)
                plt.show()

        # Print up to 5 misclassified examples
        print("\nUp to 5 misclassified examples (true → predicted):")
        for idx, (true_name, pred_name) in enumerate(wrong_examples):
            print(f"  {idx + 1}. {true_name} → {pred_name}")
        if not wrong_examples:
            print("  (No misclassifications found!)")

    def make_confusion_matrix_for_scanobject(self):

        print("Generating Confusion Matrix....")

        test_loader = self.get_test_loader()
        self.model.eval()

        all_preds = []
        all_labels = []
        test_true = []
        test_pred = []

        class_names = [
            "bag", "bin", "box", "cabinet", "chair",
            "desk", "display", "door", "shelf", "table",
            "bed", "pillow", "sink", "sofa", "toilet"
        ]

        # Will collect up to 5 wrongly classified examples:
        wrong_examples = []  # list of tuples: (true_name, predicted_name)

        with tqdm(test_loader, unit="batch") as logging_wrapper:
            logging_wrapper.set_description(f"TESTING {len(test_loader)} Batches...")

            with torch.no_grad():
                for data, label, classname in logging_wrapper:
                    # Preprocess batch exactly as training
                    (feats, coords), label = self.preprocess_test_data(data, label)

                    # Make sure feats & label are on the correct device
                    feats = feats.to(self.device)
                    label = label.to(self.device)

                    # Forward pass
                    logits = self.model(feats)
                    preds = logits.argmax(dim=1)

                    # Accumulate for confusion matrix
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(label.cpu().numpy())

                    # Check for misclassifications in this batch (up to 5 total)
                    if len(wrong_examples) < 5:
                        # Move preds & label to CPU for easy comparison
                        preds_cpu = preds.detach().cpu()
                        label_cpu = label.cpu()
                        # `classname` is a list of true class names for this batch
                        for i in range(label_cpu.size(0)):
                            if len(wrong_examples) >= 5:
                                break
                            true_idx = label_cpu[i].item()
                            pred_idx = preds_cpu[i].item()
                            if pred_idx != true_idx:
                                true_name = classname[i]  # true class name from dataset
                                pred_name = class_names[pred_idx]
                                wrong_examples.append((true_name, pred_name))

                # Concatenate everything once all batches are done
                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                # Compute and print overall metrics
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = f"Test :: test acc: {test_acc:.6f}, test avg acc: {avg_per_class_acc:.6f}"
                print(outstr)

                # Plot confusion matrix
                cm = confusion_matrix(all_labels, all_preds, labels=list(range(15)))
                plot_confusion_matrix(cm, class_names)
                plt.savefig("modelnet40_confusion_matrix.png", dpi=300)
                plt.show()

        # After looping through all batches, print up to 5 misclassified examples
        print("\nUp to 5 misclassified examples (true → predicted):")
        for idx, (true_name, pred_name) in enumerate(wrong_examples):
            print(f"  {idx + 1}. {true_name} → {pred_name}")
        if not wrong_examples:
            print("  (No misclassifications found!)")


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
                (feats, coords), label = self.preprocess_test_data(data, label)

                # ==== AMP CHANGE #4: wrap inference in autocast (optional)
                if self.device.type == 'cuda' and self.args.amp:
                    with autocast(enabled=(self.device.type == 'cuda')):
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

    def preprocess_data(self, data, label):
        if self.args.dataset == "modelnet40":
            return self.preprocess_modelnet_data(data, label)
        else:
            return self.preprocess_scanobject_data(data, label)

    def preprocess_modelnet_data(self, data, label):
        data = data.numpy()
        data = provider.random_point_dropout(data)
        data[:, :, 0:3] = provider.random_scale_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = provider.shift_point_cloud(data[:, :, 0:3])
        data = torch.Tensor(data)
        label_tensor = torch.LongTensor(label[:, 0].numpy())
        data, label = data.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True).squeeze()

        feats = data.permute(0, 2, 1)
        coords = feats[:, :, 0:3].to(self.device)
        return (feats, coords), label_tensor

    def preprocess_scanobject_data(self, data, label):
        """
        For ScanObjectNN: `data` comes in as (B, N, 6) → we split into
            feats  = (B, 6, N)   and  coords = (B, 3, N).
        """
        # 1) NumPy-side augmentations
        data_np = data.numpy()  # (B, N, C)
        data_np = provider.random_point_dropout(data_np)
        data_np[:, :, 0:3] = provider.random_scale_point_cloud(data_np[:, :, 0:3])
        data_np[:, :, 0:3] = provider.shift_point_cloud(data_np[:, :, 0:3])

        # 2) back to FloatTensor
        data_t = torch.from_numpy(data_np.astype('float32'))  # (B, N, C)

        # 3) split into feats / coords depending on dataset

        # ScanObjectNNDataset already returns (npoint, 6), so data_t is (B, N, 6)
        # feats = all 6 channels, transposed to (B, 6, N)
        feats = data_t.permute(0, 2, 1).to(self.device)  # (B, 6, N)
        # coords = first-three dims, transposed to (B, 3, N)
        coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)

        # 4) turn `label` into a 1D LongTensor of shape (B,)
        label_tensor = label.long().to(self.device)

        return (feats, coords), label_tensor

    def preprocess_test_data(self, data, label):
        if self.args.dataset == "modelnet40":
            return self.preprocess_modelnet_test_data(data, label)
        else:
            return self.preprocess_scanobject_test_data(data, label)

    def preprocess_scanobject_test_data(self, data, label):
        data_np = data.numpy()  # (B, N, C)
        data_t = torch.from_numpy(data_np.astype('float32'))  # (B, N, C)
        feats = data_t.permute(0, 2, 1).to(self.device)  # (B, 6, N)
        coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)
        label_tensor = torch.LongTensor(label).to(self.device)  # (B,)
        return (feats, coords), label_tensor

    def preprocess_modelnet_test_data(self, data, label):
        label_tensor = torch.LongTensor(label[:, 0].numpy())
        data, label = data.to(self.device), label.to(self.device).squeeze()
        feats = data.permute(0, 2, 1)
        coords = feats[:, :, 0:3].to(self.device)
        return (feats, coords), label_tensor

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

    def register_voxelization_hooks(self):
        # (A) Prepare three “empty” slots to hold each PVTConv’s voxel outputs:
        #     Each entry will be a tuple (avg_voxel_features, voxel_coords).
        #     We index them [0], [1], [2] to match the three PVTConv blocks.
        self._last_voxel_feats = [None, None, None]  # each will be shape [B, C_voxel, R, R, R]
        self._last_voxel_coords = [None, None, None]  # each will be shape [B, N_pts, 3]

        # (B) Define a small factory that creates a hook function bound to index i:
        def make_voxel_hook(i):
            def _voxel_hook(module, inp, out):
                """
                This hook is invoked right after the i-th PVTConv’s Voxelization.forward(...)
                ‘out’ is (avg_voxel_features, norm_coords). We only need index 0 of that tuple.
                """
                self._last_voxel_feats[i] = out[0]  # store [B, C_voxel, R, R, R]
                self._last_voxel_coords[i] = out[1]  # store [B, 3, N_pts]

            return _voxel_hook

        # (C) Attach one hook to each PVTConv.voxelization.  We loop over i = 0..2:
        for i in range(3):
            pvt_block = self.model.point_features[i]
            # pvt_block.voxelization is a Voxelization module
            pvt_block.voxelization.register_forward_hook(make_voxel_hook(i))

    def register_saliency_hooks(self):
        # (B) Prepare lists (or any containers) to store activations & gradients:
        self.model._attn_acts = []  # will collect forward outputs from each SparseDynamicVoxelAttention
        self.model._attn_grads = []  # will collect backward grads from each SparseDynamicVoxelAttention

        # (C) Define hook functions:
        def _forward_hook(module, inp, out):
            # `module` is the SparseDynamicVoxelAttention instance;
            # `out` is its forward result (a list of length B, each of shape [Vʼ, D]).
            self.model._attn_acts.append(out.detach().cpu())

        def _backward_hook(module, grad_in, grad_out):
            # `grad_out[0]` is the gradient of the loss w.r.t. that module’s output.
            self.model._attn_grads.append(grad_out[0].detach().cpu())

        # (D) Walk through the model, find every SparseDynamicVoxelAttention, and register:
        for name, submod in self.model.named_modules():
            if isinstance(submod, SparseDynamicVoxelAttention):
                submod.register_forward_hook(_forward_hook)
                submod.register_full_backward_hook(_backward_hook)

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

    def get_train_loader(self):
        """
        - ModelNet40: use ModelNetDataLoader
        - ScanObjectNN: use ScanObjectNNDataset (supports optional dev subset and knn_normals)
        """
        if self.args.dataset == 'modelnet40':
            return self.get_modelnet_train_loader()
        elif self.args.dataset == 'scanobjectnn':
            return self.get_scanobject_trainloader()
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

    def get_test_loader(self):
        """
        - ModelNet40: use ModelNetDataLoader (partition='test')
        - ScanObjectNN: use ScanObjectNNDataset (partition='test', optional dev subset)
        """
        if self.args.dataset == 'modelnet40':
            return self.get_modelnet_test_loader()
        else:
           return self.get_scanobject_test_loader()

    def get_modelnet_train_loader(self):
        ds = ModelNetDataset(
            npoint=self.args.num_points,
            partition='train',
            uniform=False,
            normal_channel=True,
            cache_size=15000,
            args=self.args
        )
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

    def get_scanobject_trainloader(self):

        ds = ScanObjectNNDataset(
            npoint=self.args.num_points,
            partition='train',
            args=self.args,
            knn_normals=self.args.knn_normals
        )

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

    def get_scanobject_test_loader(self):
        if self.args.dataset == 'scanobjectnn' and self.args.scanobject_compare:
            ds = ScanObjectNNDatasetModified(
                npoint=self.args.num_points,
                partition='test',
                args=self.args,
                knn_normals=self.args.knn_normals
            )

        else:
            ds = ScanObjectNNDataset(
                npoint=self.args.num_points,
                partition='test',
                args=self.args,
                knn_normals=self.args.knn_normals
            )

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

    def get_modelnet_test_loader(self):
        test_loader = DataLoader(ModelNetDataset(
            partition='test',
            npoint=self.args.num_points,
            args=self.args
        ),
            num_workers=self.args.num_workers,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.args.persist_workers,
            prefetch_factor=self.args.prefetch_factor
        )
        return test_loader

    def test_compare_with_hooks(self):
        print("[test_compare_with_hooks] → Entering method")
        self.model.eval()
        print("[test_compare_with_hooks] Model set to eval()")
        test_loader = self.get_test_loader()
        print("[test_compare_with_hooks] Obtained test_loader")

        all_results = []
        total_true = []
        total_pred = []

        for batch_idx, (data, label, classname) in enumerate(test_loader):
            print(f"\n--- Batch {batch_idx} start ---")
            # 1) Preprocess
            print(f"[Batch {batch_idx}]   Preprocessing raw data")
            (feats, coords), label = self.preprocess_test_data(data, label)
            feats = feats.to(self.device)
            coords = coords.to(self.device)
            label = label.to(self.device)
            print(f"[Batch {batch_idx}]   feats.shape={feats.shape}, coords.shape={coords.shape}, label.shape={label.shape}")

            B, _, _ = feats.shape
            print(f"[Batch {batch_idx}]   Batch size B={B}")

            # ─── Clear out old hook outputs and prior gradients ───
            print(f"[Batch {batch_idx}]   Clearing previous voxel‐hook outputs and gradients")
            for i in range(3):
                self._last_voxel_feats[i] = None
                self._last_voxel_coords[i] = None
                print(f"[Batch {batch_idx}]     Cleared _last_voxel_feats[{i}] and _last_voxel_coords[{i}]")
            self.model._attn_acts.clear()
            self.model._attn_grads.clear()
            print(f"[Batch {batch_idx}]   Cleared model._attn_acts and model._attn_grads")
            self.model.zero_grad()
            print(f"[Batch {batch_idx}]   Called model.zero_grad()")

            # ─── Forward pass through the entire network ───
            print(f"[Batch {batch_idx}]   Forward pass through model")
            logits = self.model(feats)  # each PVTConv.voxelization hook fires now
            print(f"[Batch {batch_idx}]   Received logits.shape={logits.shape}")
            preds = logits.argmax(dim=1)  # shape [B]
            print(f"[Batch {batch_idx}]   Computed preds={preds.cpu().tolist()}")
            total_true.append(label.cpu().numpy())
            total_pred.append(preds.cpu().numpy())
            print(f"[Batch {batch_idx}]   Appended batch labels/preds to total_true/total_pred")

            # At this point, each self._last_voxel_feats[0..2] should be filled in
            for stage in range(3):
                vf = self._last_voxel_feats[stage]
                vc = self._last_voxel_coords[stage]
                if vf is None or vc is None:
                    print(f"[Batch {batch_idx}]   WARNING: stage {stage} _last_voxel_feats or coords is still None!")
                else:
                    print(f"[Batch {batch_idx}]   stage {stage} voxel_feats.shape={vf.shape}, voxel_coords.shape={vc.shape}")

            # ─── Per-sample backward → grab activations/gradients ───
            for i in range(B):
                print(f"  [Batch {batch_idx}, Sample {i}] ── Starting backward for sample {i}")
                self.model.zero_grad()
                self.model._attn_acts.clear()
                self.model._attn_grads.clear()
                print(f"  [Batch {batch_idx}, Sample {i}]   Cleared grads and stored activations")

                pred_i = preds[i].item()
                print(f"  [Batch {batch_idx}, Sample {i}]   pred_i={pred_i}")
                scalar_logit = logits[i, pred_i]
                print(f"  [Batch {batch_idx}, Sample {i}]   scalar_logit obtained; calling backward()")
                scalar_logit.backward(retain_graph=True)
                print(f"  [Batch {batch_idx}, Sample {i}]   backward() complete")

                # ─── Grab each attention block’s stored activations and gradients ───
                print(f"  [Batch {batch_idx}, Sample {i}]   Fetching stored attention activations/gradients")
                a1 = self.model._attn_acts[0][i].detach().cpu()
                g1 = self.model._attn_grads[0][i].detach().cpu()
                print(f"  [Batch {batch_idx}, Sample {i}]     Stage 0: a1.shape={a1.shape}, g1.shape={g1.shape}")

                a2 = self.model._attn_acts[1][i].detach().cpu()
                g2 = self.model._attn_grads[1][i].detach().cpu()
                print(f"  [Batch {batch_idx}, Sample {i}]     Stage 1: a2.shape={a2.shape}, g2.shape={g2.shape}")

                a3 = self.model._attn_acts[2][i].detach().cpu()
                g3 = self.model._attn_grads[2][i].detach().cpu()
                print(f"  [Batch {batch_idx}, Sample {i}]     Stage 2: a3.shape={a3.shape}, g3.shape={g3.shape}")

                # ─── Build non_empty_mask for each stage ───
                print(f"  [Batch {batch_idx}, Sample {i}]   Computing non-empty masks for each resolution")
                for stage in range(3):
                    vox_feats = self._last_voxel_feats[stage][i].unsqueeze(0)
                    print(f"  [Batch {batch_idx}, Sample {i}]     Stage {stage} vox_feats.shape={vox_feats.shape}")
                    mask = modules.voxel_encoder.extract_non_empty_voxel_mask(vox_feats, self.args)
                    print(f"  [Batch {batch_idx}, Sample {i}]     Stage {stage} raw mask.shape={mask.shape}")
                    mask_1d = mask.view(-1).cpu()
                    nonzeros = mask_1d.nonzero(as_tuple=False).numel()
                    print(f"  [Batch {batch_idx}, Sample {i}]     Stage {stage} non_empty count={nonzeros}")

                # ─── Recover 3D‐center coords for each stage’s occupied voxels ───
                print(f"  [Batch {batch_idx}, Sample {i}]   Recovering 3D center coordinates for each stage")
                for stage in range(3):
                    vox_feats = self._last_voxel_feats[stage][i].unsqueeze(0)
                    Rk = vox_feats.shape[2]
                    print(f"  [Batch {batch_idx}, Sample {i}]     Stage {stage} resolution Rk={Rk}")
                    centers_k = generate_voxel_grid_centers(Rk, self.args)[0].cpu().numpy()
                    print(f"  [Batch {batch_idx}, Sample {i}]     Stage {stage} generated {centers_k.shape[0]} centers")
                    mask_1d = modules.voxel_encoder.extract_non_empty_voxel_mask(vox_feats, self.args).view(-1).cpu()
                    occ_idx = torch.nonzero(mask_1d, as_tuple=False).squeeze(1).numpy()
                    print(f"  [Batch {batch_idx}, Sample {i}]     Stage {stage} occ_idx length={len(occ_idx)}")
                    coords_occ = centers_k[occ_idx]
                    print(f"  [Batch {batch_idx}, Sample {i}]     Stage {stage} coords_occ.shape={coords_occ.shape}")

                # ─── Build result dictionary ───
                print(f"  [Batch {batch_idx}, Sample {i}]   Building result dictionary for this sample")
                item = {
                    "pred":      pred_i,
                    "true":      label[i].item(),
                    "classname": classname[i],

                    # Stage 0:
                    "coords0": coords_occ,  # from last loop, but you could store separately
                    "feat0":   a1,          # [V_occ0, C1]
                    "grad0":   g1,          # [V_occ0, C1]

                    # Stage 1:
                    "coords1": coords_occ,  # NOTE: if you want separate arrays per stage, recompute above
                    "feat1":   a2,          # [V_occ1, C2]
                    "grad1":   g2,          # [V_occ1, C2]

                    # Stage 2:
                    "coords2": coords_occ,  # same note as above
                    "feat2":   a3,          # [V_occ2, C3]
                    "grad2":   g3           # [V_occ2, C3]
                }
                all_results.append(item)
                print(f"  [Batch {batch_idx}, Sample {i}]   Appended item to all_results (len={len(all_results)})")

                # ─── Clear gradients before next sample ───
                self.model.zero_grad()
                self.model._attn_acts.clear()
                self.model._attn_grads.clear()
                print(f"  [Batch {batch_idx}, Sample {i}]   Cleared gradients/hooks for next sample")

            # ─── Compute final accuracy for this batch ───
            print(f"[Batch {batch_idx}]   Computing final accuracy so far")
            total_true_arr = np.concatenate(total_true) if total_true else np.array([])
            total_pred_arr = np.concatenate(total_pred) if total_pred else np.array([])
            if total_true_arr.size > 0:
                test_acc  = metrics.accuracy_score(total_true_arr, total_pred_arr)
                avg_class = metrics.balanced_accuracy_score(total_true_arr, total_pred_arr)
                print(f"[Batch {batch_idx}]   Test accuracy so far: acc={test_acc:.4f}, avg‐class={avg_class:.4f}")
            else:
                print(f"[Batch {batch_idx}]   No predictions yet, skipping accuracy")

            print(f"--- Batch {batch_idx} end ---")

        print("[test_compare_with_hooks] → Exiting method, returning all_results")
        return all_results

def generate_voxel_grid_centers(resolution, args):
    """
    Returns a tensor of shape (V, 3) with the 3D center coordinates
    of each voxel in a cubic grid of shape (R x R x R), normalized to [-1, 1]^3.

    Args:
        resolution (int): Number of voxels per axis (R)
        device (str or torch.device): Where to place the resulting tensor

    Returns:
        Tensor: (V, 3), where V = R^3
    """
    # Generate voxel indices in 3D grid: shape (R, R, R, 3)
    grid = torch.stack(torch.meshgrid(
        torch.arange(resolution),
        torch.arange(resolution),
        torch.arange(resolution),
        indexing='ij'  # Makes indexing (x, y, z) order
    ), dim=-1).float()  # shape: (R, R, R, 3)

    # Reshape to flat list of voxel indices: (V, 3) where V = R^3
    grid = grid.reshape(-1, 3)  # e.g., (27000, 3) for R=30

    # Compute voxel centers in normalized [-1, 1]^3 space
    grid = (grid + 0.5) / resolution  # Normalize to (0, 1)
    grid = grid * 2 - 1  # Map to (-1, 1)

    voxel_centers = grid.unsqueeze(0).expand(args.batch_size, -1, -1)  # shape: (B, R^3, 3)
    return voxel_centers.to(args.device)

