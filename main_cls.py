from __future__ import print_function
import warnings
import sys
import torch

from dynamic_config import ConfigWatcher

# This enables verbose mode for error messages in torch.
# TURN OFF IN PROD, SLOWS EVERYTHING DOWN
# torch.autograd.set_detect_anomaly(True)

from trainer import Trainer

# ignore everything
warnings.filterwarnings("ignore")
import os
import argparse
from util import IOStream

# ──────────────── IMPORTS FOR SALIENCY ─────────────────────────
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from types import SimpleNamespace
from data import ScanObjectNNDataset
from model.pvt import pvt
# ───────────────────────────────────────────────────────────────


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


# ────────────────────────── SALIENCY FUNCTION ──────────────────────────
def generate_saliency(args):
    """
    1. Force CPU + dev subset
    2. Load two pvt(...) models (DSVA vs Vanilla) on CPU
    3. Loop dev subset until DSVA is correct and Vanilla is wrong
    4. Compute two saliency vectors, plot side by side
    """

    # 1) Use GPU if available (so CUDA-based voxel ops can run)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = str(device)   # e.g. 'cuda' or 'cpu'
    args.dev_scan_subset = False   ## Setting args.dev_scan_subset = False makes it load the full ScanObjectNN dataset.



    # 2) Instantiate DSVA branch on CPU
    dsva_args = SimpleNamespace(**vars(args))
    dsva_args.use_dsva = True
    dsva_args.no_point_attention = False
    dsva_model = pvt(dsva_args).to(device)
    print(f"Loading DSVA checkpoint from {args.dsva_ckpt} ...")
    ckpt_dsva = torch.load(args.dsva_ckpt, map_location='cpu')
    dsva_model.load_state_dict(ckpt_dsva['model_state_dict'], strict=False)
    dsva_model.eval()

    # 3) Instantiate Vanilla branch on CPU
    vanilla_args = SimpleNamespace(**vars(args))
    vanilla_args.use_dsva = False
    vanilla_args.no_point_attention = False
    vanilla_model = pvt(vanilla_args).to(device)
    print(f"Loading Vanilla checkpoint from {args.vanilla_ckpt} ...")
    ckpt_vanilla = torch.load(args.vanilla_ckpt, map_location='cpu')
    vanilla_model.load_state_dict(ckpt_vanilla['model_state_dict'], strict=False)
    vanilla_model.eval()

    # 4) Build tiny dev‐subset DataLoader
    test_dataset = ScanObjectNNDataset(
        npoint=args.num_points,
        partition='test',
        args=args,
        knn_normals=getattr(args, 'knn_normals', 30)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 5) Find first sample where DSVA is correct and Vanilla is wrong
    selected_input = None
    selected_label = None

    for data, label in test_loader:
        # data shape: (1, N, 6). We need (1, 6, N).
        feats = data.permute(0, 2, 1).contiguous().to(device)  # shape: (1,6,N)
        label = label.to(device)

        # DSVA forward
        feats_dsva = feats.clone().detach().requires_grad_(True)
        with torch.no_grad():
            logits_dsva = dsva_model(feats_dsva)
        pred_dsva = logits_dsva.argmax(dim=1)

        # Vanilla forward
        feats_v = feats.clone().detach().requires_grad_(True)
        with torch.no_grad():
            logits_vanilla = vanilla_model(feats_v)
        pred_vanilla = logits_vanilla.argmax(dim=1)

        if (pred_dsva.item() == label.item()) and (pred_vanilla.item() != label.item()):
            # Found the “DSVA correct & Vanilla wrong” example
            selected_input = feats.clone().detach().requires_grad_(True)
            selected_label = label.item()
            break

    if selected_input is None:
        print("❌ No dev‐subset sample found where DSVA is correct and Vanilla is wrong.")
        sys.exit(1)

    # 6) Compute DSVA saliency (w.r.t. the true class logit)
    dsva_model.zero_grad()
    logits = dsva_model(selected_input)
    score_true = logits[0, selected_label]
    score_true.backward(retain_graph=True)
    saliency_dsva = selected_input.grad.abs().sum(dim=1).squeeze().cpu().numpy()  # shape: (N,)

    # 7) Compute Vanilla saliency (w.r.t. its predicted class logit)
    feats_v = selected_input.clone().detach().requires_grad_(True)
    vanilla_model.zero_grad()
    logits_v = vanilla_model(feats_v)
    pred_class_v = logits_v.argmax(dim=1).item()
    score_v = logits_v[0, pred_class_v]
    score_v.backward()
    saliency_vanilla = feats_v.grad.abs().sum(dim=1).squeeze().cpu().numpy()  # shape: (N,)

    # 8) Extract xyz coords (first 3 channels) for plotting
    coords = selected_input[0, :3, :].cpu().numpy().T  # (N,3)

    # 9) Plot side‐by‐side 3D saliency maps
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    p1 = ax1.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=saliency_dsva, cmap='hot', s=20
    )
    ax1.set_title('DSVA Model Saliency')
    ax1.axis('off')
    fig.colorbar(p1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122, projection='3d')
    p2 = ax2.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=saliency_vanilla, cmap='hot', s=20
    )
    ax2.set_title('Vanilla Model Saliency')
    ax2.axis('off')
    fig.colorbar(p2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    if args.saliency_savepath:
        plt.savefig(args.saliency_savepath)
    plt.show()


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
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'scanobjectnn'],
                        help='Which dataset to train/test on')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of training batch')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of test batch')
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
    parser.add_argument('--dsva_ckpt', type=str, default='', help='Path to DSVA checkpoint .pth')
    parser.add_argument('--vanilla_ckpt', type=str, default='', help='Path to Vanilla checkpoint .pth')
    parser.add_argument('--saliency_savepath', type=str, default='', help='(optional) where to save the plot')
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

    # ──────────────────── SALIENCY BRANCH ─────────────────────────
    if args.saliency:
        if not args.dsva_ckpt or not args.vanilla_ckpt:
            print("❗ You must provide both --dsva_ckpt and --vanilla_ckpt when --saliency is set.")
            sys.exit(1)
        generate_saliency(args)
        watcher.stop()
        sys.exit(0)
    # ────────────────────────────────────────────────────────────────

    _init_(args)

    if args.use_dsva:
        print("Using DSVA!")
    else:
        print("Using Window Attention!")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')  # Annoying
    set_device(args, io)
    print(f"\n{str(args)}\n")
    torch.manual_seed(args.seed)

    trainer = Trainer(args, io)

    if not args.eval:
        trainer.fit()
    elif args.scanobject_compare:
        trainer.test_compare()
    else:
        trainer.test()
 
    watcher.stop()
