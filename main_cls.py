from __future__ import print_function
import warnings
from collections import OrderedDict

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

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model/pvt.py checkpoints' + '/' + args.exp_name + '/' + 'pvt.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    train_loader = DataLoader(ModelNetDataLoader(partition='train', npoint=args.num_points), num_workers=32,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNetDataLoader(partition='test', npoint=args.num_points), num_workers=32,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)

    # Try to load models
    if args.model == 'pvt':
        model = pvt().to(device)
    else:
        raise Exception("Not implemented")

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 10, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
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
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/model.t7' % args.exp_name)

def load_model_and_fix_misspelled_keys(model, device):
    # Load the checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)

    # Fix typo in keys
    corrected_state_dict = OrderedDict(
        (k.replace("voxel_Trasformer", "voxel_Transformer"), v)
        for k, v in checkpoint.items()
    )

    model.load_state_dict(corrected_state_dict, strict=False)


def test(args, io):
    num_workers = 2
    import time

    test_loader = DataLoader(ModelNetDataLoader(partition='test', npoint=args.num_points), num_workers=num_workers,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)

    # Try to load models
    start = time.perf_counter()

    model = pvt().to(device)
    load_model_and_fix_misspelled_keys(model, device)
    end = time.perf_counter()

    print(f"Model loading took {end - start:.4f} seconds")

    model = model.eval()
    test_true = []
    test_pred = []
    first_batch_loaded = False
    with torch.no_grad():
        start = time.perf_counter()
        for data, label in test_loader:
            if not first_batch_loaded:
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

            # d) Here model(data) invokes the forward method of the PVT class
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
        io.cprint(outstr)

def set_device(args, io):
    if not args.no_cuda and torch.cuda.is_available():
        args.device = 'cuda'
        args.cuda = True
    elif torch.backends.mps.is_available():
        args.device = 'mps'
        args.cuda = False
    else:
        args.device = 'cpu'
        args.cuda = False

    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
        io.cprint(f'Using cuda')
    else:
        io.cprint(f'Using {args.device}')

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='cls', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pvt', metavar='N',
                        choices=['pvt'],
                        help='Model to use, [pvt]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.01 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='checkpoints/cls/model.t7', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')

    set_device(args, io)

    _init_()

    io.cprint(str(args))

    torch.manual_seed(args.seed)

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
