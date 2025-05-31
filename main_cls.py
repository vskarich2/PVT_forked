from __future__ import print_function
import warnings

from PVT_forked_repo.PVT_forked.trainer import Trainer

# ignore everything
warnings.filterwarnings("ignore")
import os
import argparse
import torch
from util import cal_loss, IOStream

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model/pvt.py checkpoints' + '/' + args.exp_name + '/' + 'pvt.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def test_for_cuda():
    try:
        # Try to load one extension!
        from torch.utils.cpp_extension import load
        _src_path = os.path.dirname(os.path.abspath(__file__))
        _backend = load(name='_pvt_backend',
                        extra_cflags=['-O3', '-std=c++17'],
                        sources=[os.path.join(_src_path, 'modules', 'functional', 'src', f) for f in [
                            'interpolate/neighbor_interpolate.cpp'
                        ]]
                        )
    except Exception as e:
        print(f"Could not build CUDA backend. \nFalling back to CPU stubs.")

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

    test_for_cuda()

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
    parser.add_argument('--model_path', type=str, default='checkpoints/cls/model.t7', metavar='N',
                        help='Pretrained model path')

    # Modified arguments
    parser.add_argument('--use_sgd', action='store_true',
                        help='Use SGD')
    parser.add_argument('--no_cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model')

    # Added arguments
    parser.add_argument('--use_dsva', action='store_true', help='Use DSVA')
    parser.add_argument('--use_python_fallback', action='store_true', help='Use python vs. CUDA C++ extensions.')
    parser.add_argument('--debug_verbose', action='store_true', help='Log debug.')

    args = parser.parse_args()
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    set_device(args, io)

    #_init_()
    io.cprint(str(args))

    torch.manual_seed(args.seed)
    trainer = Trainer(args, io)

    if not args.eval:
        trainer.train()
    else:
        trainer.test()
