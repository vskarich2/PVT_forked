from __future__ import print_function
from tqdm.notebook import tqdm, trange
import warnings
# ignore everything
warnings.filterwarnings("ignore")
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNetDataLoader
from model.pvt import pvt
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import provider


class Trainer():

    def __init__(self, args, io):
        self.args = args
        self.io = io
        self.device = torch.device(self.args.device)
        self.model = self.load_model(self.device)
        self.opt = self.set_optimizer(self.model)
        self.scheduler = CosineAnnealingLR(self.opt, self.args.epochs, eta_min=self.args.lr)
        self.criterion = cal_loss

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
                    logits = self.model(data)

                    preds = logits.max(dim=1)[1]
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                test_acc = metrics.accuracy_score(test_true, test_pred)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
                outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
                print(outstr)

    def train(self):

        print("\nTraining Run Starting....\n")

        train_loader = self.get_train_loader()
        test_loader = self.get_test_loader()

        best_test_acc = 0

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
            test_acc = self.test_one_epoch(epoch, test_loader, train_avg_loss)

            # Possibly save new checkpoint
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                self.save_new_checkpoint()

    def test_one_epoch(self, epoch, test_loader, train_avg_loss):

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

                logits = self.model(data)
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
        test_acc = self.check_stats(count, epoch, test_loss, test_pred, test_true, train_avg_loss)

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
            logits = self.model(data)
            loss = self.criterion(logits, label)
            curr_loss = loss.item()
            loss.backward()
            self.opt.step()

            running_loss += curr_loss
            running_count += 1.0
            running_avg = running_loss / running_count

            # 3) Push into tqdm postfix
            train_bar.set_postfix({
                "Batch Loss": f"{curr_loss:.6f}",
                "Avg. Loss": f"{running_avg:.6f}"
            })

        # Close training bar for this epoch
        train_bar.close()

        # Return final average loss value
        return (running_loss / running_count)

    def save_new_checkpoint(self):
        torch.save(
            self.model.state_dict(),
            f"checkpoints/{self.args.exp_name}/model.t7"
        )

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
    ):
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        outstr = (
            f"TrainAvgLoss={avg_train_loss:.6f} "
            f"Epoch {epoch + 1:3d}/{self.args.epochs:3d} "
            f"TestLoss={(test_loss / count):.6f} "
            f"TestAcc={test_acc:.6f} "
            f"TestAvgPerClassAcc={avg_per_class_acc:.6f}"
        )

        print(outstr)



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
            drop_last=False
        )
        return test_loader

    def get_train_loader(self):
        train_loader = DataLoader(ModelNetDataLoader(
            partition='train',
            npoint=self.args.num_points,
            args=self.args
        ),
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True
        )
        return train_loader

    def load_model(self, device):
        # Try to load models
        if self.args.model == 'pvt':
            model = pvt(args=self.args).to(device)
        else:
            raise Exception("Not implemented")
        if self.args.use_checkpoint:
            model.load_state_dict(
                torch.load(
                    self.args.model_path, map_location=device),
                strict=False)
        return model

    def set_optimizer(self, model):
        if self.args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(model.parameters(), lr=self.args.lr * 10, momentum=self.args.momentum, weight_decay=1e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        return opt
