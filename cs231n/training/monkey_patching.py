import traceback


def patch_tensor_ops_for_contiguity():
    import torch

    # Backup originals
    original_permute = torch.Tensor.permute
    original_transpose = torch.Tensor.transpose
    original_view = torch.Tensor.view

    # Patched permute
    def patched_permute(self, *dims):
        result = original_permute(self, *dims)
        if not result.is_contiguous():
            return result.contiguous()
        return result

    # Patched transpose
    def patched_transpose(self, dim0, dim1):
        result = original_transpose(self, dim0, dim1)
        if not result.is_contiguous():
            return result.contiguous()
        return result

    # Patched view
    def patched_view(self, *shape):
        if not self.is_contiguous():
            print("Non-contiguous view() called—stack trace:")
            traceback.print_stack()
            return self.contiguous().view(*shape)
        return original_view(self, *shape)



    # Apply patches
    torch.Tensor.permute = patched_permute
    torch.Tensor.transpose = patched_transpose
    torch.Tensor.view = patched_view

    print("[DEBUG PATCH] Patched Tensor.permute, Tensor.transpose, and Tensor.view for contiguous safety.")
