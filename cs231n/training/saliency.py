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
                print("GOT HERE _voxel_hook 1")
                self._last_voxel_feats[i] = out[0]  # store [B, C_voxel, R, R, R]
                self._last_voxel_coords[i] = out[1]  # store [B, 3, N_pts]

                print("GOT HERE _voxel_hook 2")
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
            outs = [out] if torch.is_tensor(out) else list(out)

            for t in outs:
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

            for batch_idx, (data, label, classname) in enumerate(test_loader):
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

                    # backward on that one logit
                    scalar_logit = out_i[0, pred_i]
                    scalar_logit.backward()
                    print("    backward complete")

                    # grab per‐stage activation/grad pairs
                    a1, g1 = self.model._attn_acts[0][0].cpu(), self.model._attn_grads[0][0].cpu()
                    a2, g2 = self.model._attn_acts[1][0].cpu(), self.model._attn_grads[1][0].cpu()
                    a3, g3 = self.model._attn_acts[2][0].cpu(), self.model._attn_grads[2][0].cpu()
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
                        centers_k = self.generate_voxel_grid_centers(Rk)[0].cpu().numpy()
                        print(f"    Centers Stage{stage} count={centers_k.shape[0]}")

                    # build and store the result dict
                    item = {
                        "pred": pred_i,
                        "true": label[i].item(),
                        "classname": classname[i],
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

    # ------------------------------------------------------------
# Example “item” dictionary (you already have this in your code):
#    - coords0, feat0, grad0  : stage 0 voxel coords, features, and gradients
#    - coords1, feat1, grad1  : stage 1 voxel coords, features, and gradients
#    - coords2, feat2, grad2  : stage 2 voxel coords, features, and gradients
#
# Here we assume:
#  • coordsN is a (V_N, 3) float‐tensor (or NumPy array) of each occupied
#    voxel’s center‐coordinates in [0,1]³ (or already normalized to your scene).
#  • featN is a (V_N, C_N) PyTorch tensor of features at that stage.
#  • gradN is a (V_N, C_N) PyTorch tensor of gradients (∂score/∂feat) at that stage.
#
# If you also have the original point‐cloud coordinates (N_pts × 3),
# you can pass them in as item['points'] so we can overlay them in white.
# ------------------------------------------------------------

    def plot_three_stage_saliency(item):
        """
        Given an `item` dict with keys:
          'coords0','feat0','grad0',
          'coords1','feat1','grad1',
          'coords2','feat2','grad2',
          (optionally) 'points'=Nx3 point‐cloud coordinates,
        compute a per‐voxel saliency score at each stage and plot 3 side‐by‐side
        3D scatterplots: voxels colored by saliency + optional overlay of raw points.
        """
        stages = [0, 1, 2]
        fig = plt.figure(figsize=(18, 6))

        for i, stage in enumerate(stages):
            ax = fig.add_subplot(1, 3, i + 1, projection='3d')

            # 1) Grab coords, feat, grad for this stage
            coords = item[f"coords{stage}"]  # (V_stage, 3) → can be torch.Tensor or np.ndarray
            feat   = item[f"feat{stage}"]    # torch.Tensor of shape (V_stage, C_stage)
            grad   = item[f"grad{stage}"]    # torch.Tensor of shape (V_stage, C_stage)

            # If coords is a torch.Tensor, convert to NumPy:
            if isinstance(coords, torch.Tensor):
                coords = coords.detach().cpu().numpy()

            # 2) Compute a simple per‐voxel saliency score.
            #    Here we do: saliency = sum over channels of |grad * feat|.
            #    You could also do saliency = ||grad||_2 or just |grad|.sum().
            with torch.no_grad():
                # Ensure feat/grad are on CPU for NumPy conversion:
                f = feat.detach().cpu()
                g = grad.detach().cpu()
                # element‐wise product → absolute → sum over channel‐dim:
                sal = (g * f).abs().sum(dim=1)      # → shape (V_stage,)
                sal = sal.cpu().numpy()

            # 3) Normalize saliency to [0,1] for nice color mapping:
            sal_min, sal_max = sal.min(), sal.max()
            if sal_max - sal_min > 1e-8:
                sal_norm = (sal - sal_min) / (sal_max - sal_min)
            else:
                sal_norm = np.zeros_like(sal)

            # 4) Plot voxels as a 3D scatter, colored by saliency:
            xs = coords[:, 0]
            ys = coords[:, 1]
            zs = coords[:, 2]
            p = ax.scatter(
                xs, ys, zs,
                c=sal_norm,
                cmap="hot",
                s=20,          # adjust marker‐size as needed
                alpha=0.8,     # semi‐transparent so points underneath can show
                edgecolors="none"
            )

            # 5) If you have the raw point cloud in item['points'], overlay it in white:
            if "points" in item:
                pts = item["points"]               # expect shape (N_pts, 3)
                if isinstance(pts, torch.Tensor):
                    pts = pts.detach().cpu().numpy()
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c="white",
                    s=1,          # tiny white dots for the point‐cloud
                    alpha=0.5     # slightly transparent so you still see voxels behind
                )

            ax.set_title(f"Stage {stage}", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        # 6) Add a single colorbar for all 3 subplots:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(p, cax=cax, label="Saliency (normalized)")

        plt.tight_layout(rect=[0, 0, 0.9, 1.0])
        plt.show()

    # ------------------------------------------------------------------
    # Example usage:
    # ------------------------------------------------------------------

    # Suppose you have already computed (in your inference loop) something like:
    #
    #   item = {
    #       "pred":      pred_i,          # (scalar)
    #       "true":      label[i].item(), # (scalar)
    #       "classname": classname[i],    # string
    #
    #       # Stage 0:
    #       "coords0": coords_occ0, # torch.Tensor of shape (V0, 3), in [0,1]^3
    #       "feat0":   a1,          # torch.Tensor of shape (V0, C1)
    #       "grad0":   g1,          # torch.Tensor of shape (V0, C1)
    #
    #       # Stage 1:
    #       "coords1": coords_occ1, # torch.Tensor of shape (V1, 3), in [0,1]^3
    #       "feat1":   a2,          # torch.Tensor of shape (V1, C2)
    #       "grad1":   g2,          # torch.Tensor of shape (V1, C2)
    #
    #       # Stage 2:
    #       "coords2": coords_occ2, # torch.Tensor of shape (V2, 3), in [0,1]^3
    #       "feat2":   a3,          # torch.Tensor of shape (V2, C3)
    #       "grad2":   g3           # torch.Tensor of shape (V2, C3)
    #   }
    #
    # (Optionally, if you also want to overlay the original N‐point point‐cloud:)
    #   item["points"] = raw_point_cloud_coords  # shape (N,3), in [0,1]^3
    #
    # Then simply call:
    #
    #   plot_three_stage_saliency(item)
    #
    # and you will see a figure with three side‐by‐side panels—“Stage 0,” “Stage 1,” “Stage 2”—where:
    #   • Each colored dot is a voxel‐coordinate, colored by the aggregated (|grad*feat|) saliency.
    #   • Any white dots (if you provided item["points"]) are the original point‐cloud overlaid semi‐transparently.
    #
    # You can of course customize:
    #   – How you compute “saliency” (maybe sum |gradients| alone, or ||gradient||₂, etc.)
    #   – The colormap (“hot”, “viridis”, “magma”, etc.)
    #   – Marker sizes (s=20, s=1) or alpha transparency.
    #   – Whether to plot all three stages or just pick one.
    # ------------------------------------------------------------------

    # =============================================================================
    # If you want to test this with dummy data (for illustration), you can do:
    # =============================================================================

