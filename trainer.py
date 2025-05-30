from __future__ import print_function
from tqdm import tqdm
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

    def test(self):
        print("Testing Run Starting....")
        import time

        test_loader = DataLoader(
            ModelNetDataLoader(
                partition='test',
                npoint=self.args.num_points),
            num_workers=self.args.num_workers,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            drop_last=False)

        device = torch.device(self.args.device)

        # Try to load models
        start = time.perf_counter()

        model = pvt(args=self.args).to(device)
        model.load_state_dict(
            torch.load(
                self.args.model_path, map_location=device),
            strict=False)

        end = time.perf_counter()

        print(f"Model loading took {end - start:.4f} seconds")

        model = model.eval()
        test_true = []
        test_pred = []
        first_batch_loaded = False
        with tqdm(test_loader, unit="batch") as logging_wrapper:
            logging_wrapper.set_description(f"TESTING {len(test_loader)} Batches...")

        with torch.no_grad():
            start = time.perf_counter()
            for data, label in logging_wrapper:
                if not first_batch_loaded:
                    print("Processing Batches....")
                    first_batch_loaded = True
                    end = time.perf_counter()
                    print(f"Lazy data loading took {end - start:.4f} seconds")

                # a) extract scalar label
                label = torch.LongTensor(label[:, 0].numpy())
                # b) move to device
                data, label = data.to(device), label.to(device).squeeze()
                # c) reshape to (B, C, N) ← originally (B, N, C)
                # (B, C, N) == (Batch, Channels, Points) == (8, 6, 1024)
                data = data.permute(0, 2, 1)

                # d) model(data) invokes the forward method of the pvt class
                logits = model(data)

                # e) take argmax over classes
                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
            self.io.cprint(outstr)

    def train(self):
        print("Training Run Starting....")
        train_loader = DataLoader(ModelNetDataLoader(
            partition='train',
            npoint=self.args.num_points),
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True)
        test_loader = DataLoader(ModelNetDataLoader(
            partition='test',
            npoint=self.args.num_points),
            num_workers=self.args.num_workers,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            drop_last=False)

        device = torch.device(self.args.device)

        # Try to load models
        if self.args.model == 'pvt':
            model = pvt().to(device)
        else:
            raise Exception("Not implemented")

        print("Let's use", torch.cuda.device_count(), "GPUs!")

        if self.args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(model.parameters(), lr=self.args.lr * 10, momentum=self.args.momentum, weight_decay=1e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        scheduler = CosineAnnealingLR(opt, self.args.epochs, eta_min=self.args.lr)
        criterion = cal_loss
        best_test_acc = 0

        for epoch in range(self.args.epochs):
            scheduler.step()
            ####################
            # Train
            ####################
            model.train()
            for data, label in train_loader:
                data = data.numpy()
                data = provider.random_point_dropout(data)
                data[:, :, 0:3] = provider.random_scale_point_cloud(data[:, :, 0:3])
                data[:, :, 0:3] = provider.shift_point_cloud(data[:, :, 0:3])
                data = torch.Tensor(data)
                label = torch.LongTensor(label[:, 0].numpy())
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                opt.zero_grad()
                logits = model(data)
                loss = criterion(logits, label)
                loss.backward()
                opt.step()

            ####################
            # Test
            ####################
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            for data, label in test_loader:
                label = torch.LongTensor(label[:, 0].numpy())
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                  test_loss * 1.0 / count,
                                                                                  test_acc,
                                                                  avg_per_class_acc)
            self.io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/model.t7' % self.args.exp_name)
