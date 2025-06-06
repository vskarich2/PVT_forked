import torch

import provider


class DataPreprocessingMixin:
    """
    Mixin providing methods to preprocess data
    """
    def preprocess_data(self, data, label):
        if self.args.dataset == "modelnet40":
            return self.preprocess_modelnet_data(data, label)
        else:
            return self.preprocess_scanobject_data(data, label)


    def preprocess_modelnet_data(self, data, label):
        data = data.numpy()
        data = provider.random_point_dropout(data)
        data[:, :, 0:3] = provider.random_scale_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = provider.shift_point_cloud(data[:, :, 0:3])
        data = torch.Tensor(data)
        label_tensor = torch.LongTensor(label[:, 0].numpy())
        data, label = data.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True).squeeze()

        feats = data.permute(0, 2, 1)
        coords = feats[:, :, 0:3].to(self.device)
        return (feats, coords), label_tensor


    def preprocess_scanobject_data(self, data, label):
        """
        For ScanObjectNN: `data` comes in as (B, N, 6) → we split into
            feats  = (B, 6, N)   and  coords = (B, 3, N).
        """
        # 1) NumPy-side augmentations
        data_np = data.numpy()  # (B, N, C)
        data_np = provider.random_point_dropout(data_np)
        data_np[:, :, 0:3] = provider.random_scale_point_cloud(data_np[:, :, 0:3])
        data_np[:, :, 0:3] = provider.shift_point_cloud(data_np[:, :, 0:3])

        # 2) back to FloatTensor
        data_t = torch.from_numpy(data_np.astype('float32'))  # (B, N, C)

        # 3) split into feats / coords depending on dataset

        # ScanObjectNNDataset already returns (npoint, 6), so data_t is (B, N, 6)
        # feats = all 6 channels, transposed to (B, 6, N)
        feats = data_t.permute(0, 2, 1).to(self.device)  # (B, 6, N)
        # coords = first-three dims, transposed to (B, 3, N)
        coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)

        # 4) turn `label` into a 1D LongTensor of shape (B,)
        label_tensor = label.long().to(self.device)

        return (feats, coords), label_tensor


    def preprocess_test_data(self, data, label):
        if self.args.dataset == "modelnet40":
            return self.preprocess_modelnet_test_data(data, label)
        else:
            return self.preprocess_scanobject_test_data(data, label)


    def preprocess_scanobject_test_data(self, data, label):
        data_np = data.numpy()  # (B, N, C)
        data_t = torch.from_numpy(data_np.astype('float32'))  # (B, N, C)
        feats = data_t.permute(0, 2, 1).to(self.device)  # (B, 6, N)
        coords = data_t[:, :, 0:3].permute(0, 2, 1).to(self.device)  # (B, 3, N)
        label_tensor = torch.LongTensor(label).to(self.device)  # (B,)
        return (feats, coords), label_tensor


    def preprocess_modelnet_test_data(self, data, label):
        label_tensor = torch.LongTensor(label[:, 0].numpy())
        data, label = data.to(self.device), label.to(self.device).squeeze()
        feats = data.permute(0, 2, 1)
        coords = feats[:, :, 0:3].to(self.device)
        return (feats, coords), label_tensor