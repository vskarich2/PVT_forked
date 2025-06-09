import numpy as np
import sklearn.metrics as metrics
import torch
from matplotlib import pyplot as plt

import modules
from cs231n.dsva.dsva_cross_attention import SparseDynamicVoxelAttention
from cs231n.training.voxel_grid_centers import VoxelGridCentersMixin


class SaliencyMixin(VoxelGridCentersMixin):
    """
    Mixin providing methods to create saliency functions
    """
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
        # prepare our buffers
        self.model._attn_acts = []   # list of activation tensors
        self.model._attn_grads = []  # list of gradient tensors

        def _forward_hook(module, inp, out):
            # normalize to a list of tensors

            for t in out:
                # 1) save the activation
                self.model._attn_acts.append(t.detach().cpu())

                # 2) register a hook on THAT tensor to grab its grad on backward
                def _grab_grad(grad, idx=len(self.model._attn_grads)):
                    # this will run during .backward()
                    self.model._attn_grads.insert(idx, grad.detach().cpu())

                t.register_hook(_grab_grad)



        # walk and attach only the forward hooks
        for name, submod in self.model.named_modules():
            if isinstance(submod, SparseDynamicVoxelAttention):
                submod.register_forward_hook(_forward_hook)

        # no module‐level backward hooks needed any more

    def test_compare_with_hooks(self):
        print("[test_compare_with_hooks] → Entering method")
        self.model.eval()
        test_loader = self.get_test_loader()

        all_results = []
        total_true = []
        total_pred = []
        ran_once = False

        for batch_idx, (data, label, classname) in enumerate(test_loader):
            if ran_once:
                break
            ran_once = True

            # Save original point cloud for overlay
            # data: Tensor[B, N, D] or similar
            orig_data = data.detach().cpu().clone()  # keep on CPU

            print(f"\n--- Batch {batch_idx} start ---")
            (feats, coords), label = self.preprocess_test_data(data, label)
            feats, coords, label = (
                feats.to(self.device),
                coords.to(self.device),
                label.to(self.device)
            )
            B, C_in, N = feats.shape
            print(f"[Batch {batch_idx}] shapes: feats={feats.shape}, coords={coords.shape}, label={label.shape}")

            # ─── Clear previous hooks & grads ───
            for s in range(3):
                self._last_voxel_feats[s] = None
                self._last_voxel_coords[s] = None
            self.model._attn_acts.clear()
            self.model._attn_grads.clear()
            self.model.zero_grad()

            # ─── Batch‐level forward ───
            print(f"[Batch {batch_idx}] Forwarding full batch")
            logits = self.model(feats)
            preds = logits.argmax(dim=1)
            total_true.append(label.cpu().numpy())
            total_pred.append(preds.cpu().numpy())

            # ─── Stash batch‐level voxel feats/coords ───
            batch_voxel_feats = []
            batch_voxel_coords = []
            for stage in range(3):
                vf = self._last_voxel_feats[stage]
                vc = self._last_voxel_coords[stage]
                if vf is None or vc is None:
                    raise RuntimeError(f"Stage {stage} voxel hook missing!")
                print(f"[Batch {batch_idx}] Stage {stage} voxel_feats={vf.shape}, voxel_coords={vc.shape}")

                batch_voxel_feats.append(vf.detach().cpu().clone())
                batch_voxel_coords.append(vc.detach().cpu().clone())

            # ─── Per‐sample backward & saliency ───
            for i in range(B):
                print(f"  [Batch {batch_idx}, Sample {i}] single‐sample pass")
                # clear before each sample
                self.model.zero_grad()
                self.model._attn_acts.clear()
                self.model._attn_grads.clear()

                # single‐sample forward → hooks fire again for saliency
                feat_i = feats[i: i + 1].detach().requires_grad_(True)
                out_i = self.model(feat_i)
                pred_i = out_i.argmax(dim=1).item()
                print(f"    pred={pred_i}")

                a1  = self.model._attn_acts[0].cpu()
                a2  = self.model._attn_acts[1].cpu()
                a3  = self.model._attn_acts[2].cpu()

                # backward on that one logit
                scalar_logit = out_i[0, pred_i]
                scalar_logit.backward()
                print("    backward complete")

                #grab per‐stage activation/grad pairs
                g1 =  self.model._attn_grads[0].cpu()
                g2 =  self.model._attn_grads[1].cpu()
                g3 =  self.model._attn_grads[2].cpu()
                print(f"    Stage0 a1={a1.shape}, g1={g1.shape}")
                print(f"    Stage1 a2={a2.shape}, g2={g2.shape}")
                print(f"    Stage2 a3={a3.shape}, g3={g3.shape}")

                # compute non‐empty masks using the STASHED batch feats
                for stage in range(3):
                    vox_feats = batch_voxel_feats[stage][i: i + 1]  # (1, C, R, R, R)
                    mask = modules.voxel_encoder.extract_non_empty_voxel_mask(vox_feats, self.args)
                    print(f"    Mask Stage{stage}={mask.shape}")

                # recover centers once more
                for stage in range(3):
                    Rk = batch_voxel_feats[stage].shape[2]
                    # Compute voxel centers in normalized [-1, 1]^3 space
                    centers_k = self.generate_voxel_grid_centers(Rk)[0].cpu().numpy()
                    print(f"    Centers Stage{stage} count={centers_k.shape[0]}")

                # build and store the result dict, now including original pointcloud
                item = {
                    "pred": pred_i,
                    "true": label[i].item(),
                    "classname": classname[i],
                    "pointcloud": orig_data[i].numpy(),
                    "coords0": batch_voxel_coords[0][i].numpy(),
                    "feat0": a1,
                    "grad0": g1,
                    "coords1": batch_voxel_coords[1][i].numpy(),
                    "feat1": a2,
                    "grad1": g2,
                    "coords2": batch_voxel_coords[2][i].numpy(),
                    "feat2": a3,
                    "grad2": g3,
                }
                all_results.append(item)

            print(f"--- Batch {batch_idx} end ---")

        print("[test_compare_with_hooks] → Exiting method")
        return all_results

    def inspect_saliency_results(self, item):
        """
        For each entry in results, print out:
          - pred / true / classname types
          - for each stage 0,1,2: coords shape & dtype (should be numpy floats),
                                feat shape & dtype (torch.FloatTensor),
                                grad shape & dtype (torch.FloatTensor)
        """

        print(f" pred : {item['pred']}   (type={type(item['pred'])})")
        print(f" true : {item['true']}   (type={type(item['true'])})")
        print(f" classname : {item['classname']}   (type={type(item['classname'])})")
        for stage in (0, 1, 2):
            coords = item[f"coords{stage}"]
            feat = item[f"feat{stage}"]
            grad = item[f"grad{stage}"]
            print(f" Stage {stage}:")
            print(f"   coords{stage}.shape = {coords.shape}, dtype = {coords.dtype}")
            print(f"   feat{stage}.shape   = {tuple(feat.shape)}, dtype = {feat.dtype}")
            print(f"   grad{stage}.shape   = {tuple(grad.shape)}, dtype = {grad.dtype}")

    def plot_three_stage_saliency(self,
                                  item,
                                  voxel_cmap: str = "viridis",
                                  point_color: str = "#888888",
                                  elev: float = 20,
                                  azim: float = 45,
                                  figsize=(18, 6)):
        """
        Given an `item` dict with keys:
          'coords0','feat0','grad0',
          'coords1','feat1','grad1',
          'coords2','feat2','grad2',
          'pointcloud'=N×D raw points (optional),
        compute per‑voxel saliency = sum(|feat*grad|) and
        plot 3 side‑by‑side 3D scatterplots:
          • Voxels colored by normalized saliency
          • Raw point cloud overlaid in gray
        """
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        stages = [0, 1, 2]
        fig = plt.figure(figsize=figsize)

        # 1) Gather coords for global bounds
        all_xyz = []
        for stage in stages:
            c = item[f"coords{stage}"]
            if torch.is_tensor(c):
                c = c.cpu().numpy()
            # transpose if stored as (3, V)
            if isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[0] == 3 and c.shape[1] != 3:
                c = c.T
            all_xyz.append(c)

        # 2) Include raw point cloud (XYZ only)
        if "pointcloud" in item:
            pts = item["pointcloud"]
            if torch.is_tensor(pts):
                pts = pts.cpu().numpy()
            if pts.ndim == 2 and pts.shape[1] > 3:
                pts = pts[:, :3]
            all_xyz.append(pts)

        # 3) Compute global spatial bounds
        all_xyz = np.concatenate(all_xyz, axis=0)
        xyz_min, xyz_max = all_xyz.min(axis=0), all_xyz.max(axis=0)

        # 4) Plot each stage
        for i, stage in enumerate(stages):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")

            # fetch & normalize coords
            coords = item[f"coords{stage}"]
            if torch.is_tensor(coords):
                coords = coords.cpu().numpy()
            if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[0] == 3 and coords.shape[1] != 3:
                coords = coords.T

            # fetch feats & grads
            feat = item[f"feat{stage}"]
            grad = item[f"grad{stage}"]
            if not torch.is_tensor(feat):
                feat = torch.as_tensor(feat)
            if not torch.is_tensor(grad):
                grad = torch.as_tensor(grad)

            # 4a) Align features to voxel axis
            V = coords.shape[0]
            if feat.dim() == 2 and feat.shape[1] == V and feat.shape[0] != V:
                feat = feat.T
                grad = grad.T

            # 4b) Compute saliency per voxel
            p = (feat.cpu() * grad.cpu()).abs()
            if p.dim() == 1:
                sal = p.numpy()
            else:
                sal = p.sum(dim=1).numpy()
            sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)

            # scatter voxels colored by saliency
            ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=sal_norm,
                cmap=voxel_cmap,
                s=25,
                alpha=0.8,
                edgecolor="none"
            )

            # overlay raw point cloud
            if "pointcloud" in item:
                pts_overlay = item["pointcloud"]
                if torch.is_tensor(pts_overlay):
                    pts_overlay = pts_overlay.cpu().numpy()
                if pts_overlay.ndim == 2 and pts_overlay.shape[1] > 3:
                    pts_overlay = pts_overlay[:, :3]
                ax.scatter(
                    pts_overlay[:, 0], pts_overlay[:, 1], pts_overlay[:, 2],
                    c=point_color,
                    s=1,
                    alpha=0.4
                )

            # formatting
            ax.set_title(f"Stage {stage}", fontsize=14)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_xlim(xyz_min[0], xyz_max[0])
            ax.set_ylim(xyz_min[1], xyz_max[1])
            ax.set_zlim(xyz_min[2], xyz_max[2])
            ax.view_init(elev=elev, azim=azim)

        # 5) Global colorbar
        m = plt.cm.ScalarMappable(cmap=voxel_cmap)
        m.set_array([])
        cbar = fig.colorbar(m, ax=fig.axes, fraction=0.02, pad=0.04)
        cbar.set_label("Saliency (normalized)", fontsize=12)

        # 6) Suptitle
        fig.suptitle(
            f"3-Stage Saliency & Pointcloud Overlay\n  (pred={item['pred']}  true={item['true']}  {item['classname']})",
            fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 0.95, 0.92])
        plt.show()
