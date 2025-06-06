from __future__ import print_function

import torch
from torch.utils.data import DataLoader

from data import ScanObjectNNDatasetModified, ScanObjectNNDataset, ModelNetDataset

# ignore everything

torch.backends.cudnn.benchmark = True

class DataLoaderMixin:
    """
    Mixin providing methods to initialize dataloaders
    """

    def get_train_loader(self):
        """
        - ModelNet40: use ModelNetDataLoader
        - ScanObjectNN: use ScanObjectNNDataset (supports optional dev subset and knn_normals)
        """
        if self.args.dataset == 'modelnet40':
            return self.get_modelnet_train_loader()
        elif self.args.dataset == 'scanobjectnn':
            return self.get_scanobject_trainloader()
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")


    def get_test_loader(self):
        """
        - ModelNet40: use ModelNetDataLoader (partition='test')
        - ScanObjectNN: use ScanObjectNNDataset (partition='test', optional dev subset)
        """
        if self.args.dataset == 'modelnet40':
            return self.get_modelnet_test_loader()
        else:
            return self.get_scanobject_test_loader()


    def get_modelnet_train_loader(self):
        ds = ModelNetDataset(
            npoint=self.args.num_points,
            partition='train',
            uniform=False,
            normal_channel=True,
            cache_size=15000,
            args=self.args
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.args.persist_workers,
            prefetch_factor=self.args.prefetch_factor,
            num_workers=self.args.num_workers
        )


    def get_scanobject_trainloader(self):
        ds = ScanObjectNNDataset(
            npoint=self.args.num_points,
            partition='train',
            args=self.args,
            knn_normals=self.args.knn_normals
        )

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.args.persist_workers,
            prefetch_factor=self.args.prefetch_factor,
            num_workers=self.args.num_workers
        )


    def get_scanobject_test_loader(self):
        if self.args.dataset == 'scanobjectnn' and self.args.scanobject_compare:
            ds = ScanObjectNNDatasetModified(
                npoint=self.args.num_points,
                partition='test',
                args=self.args,
                knn_normals=self.args.knn_normals
            )

        else:
            ds = ScanObjectNNDataset(
                npoint=self.args.num_points,
                partition='test',
                args=self.args,
                knn_normals=self.args.knn_normals
            )

        return DataLoader(
            ds,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.args.persist_workers,
            prefetch_factor=self.args.prefetch_factor
        )


    def get_modelnet_test_loader(self):
        test_loader = DataLoader(ModelNetDataset(
            partition='test',
            npoint=self.args.num_points,
            args=self.args
        ),
            num_workers=self.args.num_workers,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.args.persist_workers,
            prefetch_factor=self.args.prefetch_factor
        )
        return test_loader