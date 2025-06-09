from __future__ import print_function

import sys
import warnings

import torch

from cs231n.training.trainer import Trainer
from dynamic_config import ConfigWatcher

# This enables verbose mode for error messages in torch.
# TURN OFF IN PROD, SLOWS EVERYTHING DOWN
# torch.autograd.set_detect_anomaly(True)

# ignore everything
warnings.filterwarnings("ignore")
import os
import argparse
from util import IOStream

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    os.system('cp model/pvt.py checkpoints' + '/' + args.exp_name + '/' + 'pvt.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def print_os():
    if sys.platform == 'win32':
        print("This is a Windows machine.")
    elif sys.platform == 'darwin':
        print("This is a macOS machine.")
    else:
        print(f"This is: {sys.platform}")


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

    print_os()

    if args.cuda:
        print(
            'Using GPU : id:' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' device')
        torch.cuda.manual_seed(args.seed)
        print(f'Using cuda')
    else:
        print(f'Using {args.device}')

if __name__ == "__main__":
    # Training settings
    watcher = ConfigWatcher("prod_config.yaml", reload_interval=10.0)
    watcher.start()

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='cls', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pvt', metavar='N',
                        choices=['pvt'],
                        help='Model to use, [pvt]')
    #parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',choices=['modelnet40'])
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'scanobjectnn'],
                        help='Which dataset to train/test on') #Accomodate new dataset scanobjectnn
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of training batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of test batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.01 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of threads to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--knn_normals', type=int, default=30,
                        help='knn normals value')

    # Modified arguments
    parser.add_argument('--model_path', type=str, default='None', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--use_sgd', action='store_true',
                        help='Use SGD')
    parser.add_argument('--no_cuda', action='store_true',
                        help='if --no_cuda is set, model will run on CPU.')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model')


    # Added arguments
    parser.add_argument('--amp', action='store_true', help='Use amp speedup.')
    parser.add_argument('--dev_scan_subset', action='store_true', help='Use dev scanobject.')
    parser.add_argument('--persist_workers', action='store_true', help='Persist workers between epochs.')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='How much data to prefetch')

    parser.add_argument('--drop_path', type=int, default=1,
                        help='1 or 2 for attention dropout')

    parser.add_argument('--knn_size_fine', type=int, default=10,
                        help='Number of total neighbors to use in KNN.')

    parser.add_argument('--top_k_select_fine', type=int, default=4,
                        help='Number of top neighbors to use in sparse attention.')


    parser.add_argument('--knn_size_coarse', type=int, default=10,
                        help='Number of total neighbors to use in KNN.')

    parser.add_argument('--top_k_select_coarse', type=int, default=4,
                        help='Number of top neighbors to use in sparse attention.')

    parser.add_argument('--no_point_attention', action='store_true', help='Use window attention')

    parser.add_argument('--eight_heads', action='store_true', help='8 heads for coarse attention')

    parser.add_argument('--large_attn', action='store_true', help='Use window attention')

    parser.add_argument('--scanobject_compare', action='store_true', help='Use window attention')

    parser.add_argument('--saliency', action='store_true', help='Generate saliency map')
    parser.add_argument('--conf_matrix', action='store_true', help='Generate confusion matrix')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--use_dsva', action='store_true', help='Use DSVA')
    parser.add_argument('--use_python_fallback', action='store_true', help='Use python vs. CUDA C++ extensions.')
    parser.add_argument('--debug_verbose', action='store_true', help='Log debug.')
    parser.add_argument('--local_dev', action='store_true', help='Use dev dataset plus other dev-only things.')
    parser.add_argument('--use_checkpoint', action='store_true', help='Load pretrained model from checkpoint. '
                                                                      'If this flag is set, the --model_path flag must be set to the correct checkpoint')
    args = parser.parse_args()

    if not args.local_dev:
        _init_(args)

    if args.use_dsva:
        print("Using DSVA!")
    else:
        print("Using Window Attention!")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log') # Annoying
    set_device(args, io)
    print(f"\n{str(args)}\n")
    torch.manual_seed(args.seed)

    trainer = Trainer(args, io)

    if not args.eval:
        trainer.fit()
    elif args.conf_matrix and args.dataset == "modelnet40" and args.eval:
        print("Making confusion matrix for modelnet40")
        trainer.make_confusion_matrix_for_modelnet()
    elif args.conf_matrix and args.dataset == "scanobjectnn" and args.eval:
        print("Making confusion matrix for scanobject")
        trainer.make_confusion_matrix_for_scanobject()
    elif args.scanobject_compare and args.dataset == "scanobjectnn" and args.eval:
        # print("Running saliency map data logic for scanobject")
        # import pickle
        # all_results = pickle.load(open('/Users/vskarich/cs231n_final_project/PVT_forked_repo/PVT_forked/saliency_results.pkl', 'rb'))
        #trainer.plot_three_stage_saliency(all_results[0])  # etc.
        trainer.test_compare_with_hooks()

    else:
        print("Running stand-alone test!!")
        trainer.stand_alone_test()

    watcher.stop()