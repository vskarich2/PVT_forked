import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSparseSaliency:
    """
    Compute three types of saliency maps for a dynamic sparse voxel‐attention model:
      1. Attention flow (aggregated sparse attention weights)
      2. Gradient sensitivity (gradient × input)
      3. Local occlusion perturbations (voxel masking)

    Assumes:
    - `model` is a PyTorch nn.Module implementing dynamic sparse voxel attention.
    - During forward, each attention block stores its sparse attention weights (for each anchor)
      in `module.attn_weights`, a tensor of shape [M, k] or [M, M] (dense), where:
        * M = number of non‐empty voxels in the current input
        * k = number of neighbors per anchor (small constant)
      If your attention modules do not natively store weights, you may need to modify them or
      register forward hooks to capture the attention scores.
    - The model’s forward(...) method returns raw logits of shape [1, num_classes] (for batch size 1).
    - The input to this saliency class is the voxel‐feature tensor X of shape [M, D], where
      M = number of non‐empty voxels and D = feature dimension. We add a batch dimension
      so that the model sees [1, M, D].
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize with a trained dynamic sparse attention model.
        If the model’s attention submodules do not automatically store their weights,
        you can register forward hooks on them to populate `self.attention_maps`.
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

        # Container to hold attention weights from each block/layer
        # Each entry in self.attention_maps should be a Tensor of shape [M, k] or [M, M].
        self.attention_maps = []

        # Attempt to register hooks on all modules named "attention" or instance of a custom Attention class.
        # Modify this list to match your actual attention module classes.
        for module in self.model.modules():
            if hasattr(module, 'attn_weights') or module.__class__.__name__.lower().endswith("attention"):
                module.register_forward_hook(self._capture_attention_hook)

    def _capture_attention_hook(self, module, input, output):
        """
        Forward hook to capture attention weights from modules that store them during forward.
        Assumes each attention module sets `self.attn_weights` internally (shape [M, k] or [M, M]).
        If your attention module uses a different attribute, adjust accordingly.
        """
        if hasattr(module, 'attn_weights'):
            # Clone to detach from computation graph
            attn = module.attn_weights.detach().cpu()
            self.attention_maps.append(attn)

    def clear_attention(self):
        """Clear stored attention maps before a new forward pass."""
        self.attention_maps = []

    def compute_attention_flow(self, voxel_feats: torch.Tensor, normalize: bool = True):
        """
        Compute attention‐flow saliency for each voxel by aggregating attention weights
        from all captured attention maps.

        Args:
            voxel_feats (Tensor): shape [M, D], no batch dimension.
            normalize (bool): if True, normalize final scores to [0, 1].

        Returns:
            attn_saliency (Tensor): shape [M], aggregated importance per voxel.
                If each attention map is [M, k], we sum incoming edges per voxel.
                If attention map is [M, M], we sum over rows/columns appropriately.
        """
        # Prepare input
        X = voxel_feats.unsqueeze(0).to(self.device)  # shape [1, M, D]
        M, D = X.shape[1], X.shape[2]

        # Run a clean forward to populate self.attention_maps
        self.clear_attention()
        with torch.no_grad():
            _ = self.model(X)  # logits; we don't need the output here

        if not self.attention_maps:
            raise RuntimeError("No attention maps captured. "
                               "Ensure your attention modules store `attn_weights` during forward.")

        # Initialize aggregated importance per voxel
        agg_importance = torch.zeros(M, dtype=torch.float32)

        for attn in self.attention_maps:
            # attn: either shape [M, k] (sparse neighbor weights) or [M, M] (dense)
            if attn.dim() == 2 and attn.size(1) != M:
                # Sparse attention: attn[i, :] = weights from anchor i to its k neighbors.
                # We need neighbor indices; assume module also stored them as `module.neighbor_idx`
                # of shape [M, k]. If not, user must adapt accordingly.
                if not hasattr(module, 'neighbor_idx'):
                    raise RuntimeError("Sparse attention map detected, but no `neighbor_idx` available "
                                       "to map weights back to voxel indices.")
                neighbor_idx = module.neighbor_idx.detach().cpu()  # shape [M, k]
                # Sum each weight into its target voxel j
                for i in range(M):
                    for idx_pos in range(attn.size(1)):
                        j = neighbor_idx[i, idx_pos].item()
                        agg_importance[j] += attn[i, idx_pos].item()
            else:
                # Dense attention: simply sum over incoming edges; attn[i, j] = weight from i->j
                # We sum columns of attn
                # attn should be shape [M, M]
                agg_importance += attn.sum(dim=0)

        if normalize:
            agg_importance = (agg_importance - agg_importance.min()) / (
                    agg_importance.max() - agg_importance.min() + 1e-10
            )
        return agg_importance  # shape [M], on CPU

    def compute_gradient_sensitivity(self, voxel_feats: torch.Tensor, target_class: int, norm_type: str = 'l2'):
        """
        Compute gradient sensivity saliency: |x * grad_x| or norm(grad_x).

        Args:
            voxel_feats (Tensor): shape [M, D], no batch dim.
            target_class (int): the index of the logit to backprop from.
            norm_type (str): 'l2' or 'linf' norm over D channels.

        Returns:
            grad_saliency (Tensor): shape [M], importance per voxel.
        """
        X = voxel_feats.unsqueeze(0).to(self.device).requires_grad_(True)  # [1, M, D]
        logits = self.model(X)  # [1, num_classes]
        score = logits[0, target_class]
        # Backprop
        self.model.zero_grad()
        score.backward(retain_graph=False)
        grads = X.grad.detach().cpu().squeeze(0)  # shape [M, D]

        if norm_type == 'l2':
            saliency = torch.norm(voxel_feats * grads, p=2, dim=1)  # elementwise × then L2
        elif norm_type == 'linf':
            saliency = torch.max(torch.abs(voxel_feats * grads), dim=1)[0]
        else:
            raise ValueError("Unsupported norm_type: choose 'l2' or 'linf'")

        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        return saliency  # shape [M]

    def compute_occlusion_saliency(self, voxel_feats: torch.Tensor, target_class: int, baseline: float = 0.0):
        """
        Compute occlusion‐based saliency by zeroing out each voxel's feature vector and
        measuring the drop in the target logit.

        Args:
            voxel_feats (Tensor): shape [M, D], no batch dim.
            target_class (int): index of logit to measure.
            baseline (float): value to assign to each dropped voxel (default = 0).

        Returns:
            occ_saliency (Tensor): shape [M], where occ_saliency[j] =
                original_score - score_with_voxel_j_masked.
        """
        device = self.device
        M, D = voxel_feats.shape
        X_base = voxel_feats.unsqueeze(0).to(device)  # [1, M, D]

        # Compute original score
        with torch.no_grad():
            orig_logits = self.model(X_base)
            orig_score = orig_logits[0, target_class].item()

        occ_saliency = torch.zeros(M, dtype=torch.float32)

        # For each voxel j, zero out its feature vector and measure score drop
        for j in range(M):
            X_perturbed = X_base.clone()
            X_perturbed[0, j, :] = baseline  # mask voxel j
            with torch.no_grad():
                pert_logits = self.model(X_perturbed)
                pert_score = pert_logits[0, target_class].item()
            occ_saliency[j] = orig_score - pert_score

        # Some occlusion scores could be negative if masking accidentally increases confidence;
        # clamp to zero so that negative "importance" becomes zero.
        occ_saliency = torch.clamp(occ_saliency, min=0.0)

        # Normalize to [0, 1]
        occ_saliency = (occ_saliency - occ_saliency.min()) / (occ_saliency.max() - occ_saliency.min() + 1e-10)
        return occ_saliency  # shape [M]

    def compute_all_saliency(self, voxel_feats: torch.Tensor, target_class: int):
        """
        Convenience method to compute all three saliency maps and return them in a dict.

        Args:
            voxel_feats (Tensor): shape [M, D], no batch dim.
            target_class (int): index of target class.

        Returns:
            dict with keys {
                'attention_flow': Tensor[M],
                'gradient_sensitivity': Tensor[M],
                'occlusion': Tensor[M]
            }
        """
        atn_flow = self.compute_attention_flow(voxel_feats, normalize=True)
        grad_sens = self.compute_gradient_sensitivity(voxel_feats, target_class, norm_type='l2')
        occ = self.compute_occlusion_saliency(voxel_feats, target_class)
        return {
            'attention_flow': atn_flow,
            'gradient_sensitivity': grad_sens,
            'occlusion': occ
        }
