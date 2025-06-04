import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import open3d as o3d

def pc_normalize(pc):
    """Normalize point cloud to unit sphere
    Args:
        pc: point cloud [N, D]
    Returns:
        normalized point cloud centered at origin with max radius 1
    """
    centroid = np.mean(pc, axis=0)  # Compute center of mass
    pc = pc - centroid              # Center the point cloud
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # Get max distance from center
    pc = pc / m                     # Scale to unit sphere
    return pc

def farthest_point_sample(xyz, npoint):
    """Sample points uniformly in 3D space using farthest point sampling
    Args:
        xyz: point cloud [B, N, 3]
        npoint: number of points to sample
    Returns:
        centroids: sampled point indices [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10  # Initialize distances to infinity
    # Randomly choose first point
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest  # Add point to results
        # Get xyz coordinates of current farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # Compute distances from this point to all others
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update minimum distances
        distance = torch.min(distance, dist)
        # Get point with maximum distance as next center
        farthest = torch.max(distance, -1)[1]
    return centroids

class ModelNetDataLoader(Dataset):
    """DataLoader for ModelNet40 dataset"""
    def __init__(
            self,
            npoint=1024,
            partition='train',
            uniform=False,
            normal_channel=True,
            cache_size=15000,
            args=None
    ):
        """
        Args:
            npoint: Number of points to sample
            partition: 'train' or 'test'
            uniform: Whether to use uniform sampling
            normal_channel: Whether to include normal features
            cache_size: How many data points to cache in memory
        """
        self.args = args
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        if self.args.local_dev:
            DATA_DIR = os.path.join(BASE_DIR, 'data', 'dev_modelnet40_normal_resampled')
            print(f"{partition.upper()}: Using dataset: dev_modelnet40_normal_resampled")
        else:
            DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled')
            print(f"{partition.upper()}: Using dataset: modelnet40_normal_resampled")

        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(DATA_DIR, 'modelnet40_shape_names.txt')
        
        # Read category names and create mapping
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel
        
        # Get file paths for train/test split
        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(DATA_DIR, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(DATA_DIR, 'modelnet40_test.txt'))]
        
        assert (partition == 'train' or partition == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[partition]]
        
        # List of (shape_name, shape_txt_file_path) tuples
        self.datapath = [(shape_names[i], os.path.join(DATA_DIR, shape_names[i], shape_ids[partition][i]) + '.txt') 
                         for i in range(len(shape_ids[partition]))]
        
        print('The size of %s data is %d'%(partition, len(self.datapath)))
        
        self.cache_size = cache_size
        self.cache = {}  # Cache for loaded point clouds

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        """Get a single item from dataset with caching"""
        if index in self.cache:
            point_set, class_id = self.cache[index]
        else:
            fn = self.datapath[index]
            class_id = self.classes[self.datapath[index][0]]  # Get class ID
            class_id = np.array([class_id]).astype(np.int32)
            
            # Load point cloud
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            
            # Sample points
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

            # Normalize coordinates to unit sphere
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            
            # Remove surface normal channels if not needed
            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            # Cache if there's space
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, class_id)

        return point_set, class_id

    def __getitem__(self, index):
        return self._get_item(index)

def load_data_cls(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

class _ShapeNetDataset(Dataset):
    def __init__(self, num_points, partition='trainval', with_normal=True, with_one_hot_shape_id=True,
                 normalize=True, jitter=True):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.join(BASE_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
        self.num_points = num_points
        self.split = partition
        self.with_normal = with_normal
        self.with_one_hot_shape_id = with_one_hot_shape_id
        self.normalize = normalize
        if partition == 'trainval':
            self.jitter = jitter
        else:
            self.jitter = False

        shape_dir_to_shape_id = {}
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for shape_id, line in enumerate(f):
                shape_name, shape_dir = line.strip().split()
                shape_dir_to_shape_id[shape_dir] = shape_id
        file_paths = []
        if self.split == 'trainval':
            split = ['train', 'val']
        else:
            split = ['test']
        for s in split:
            with open(os.path.join(self.root, 'train_test_split', f'shuffled_{s}_file_list.json'), 'r') as f:
                file_list = json.load(f)
                for file_path in file_list:
                    _, shape_dir, filename = file_path.split('/')
                    file_paths.append(
                        (os.path.join(self.root, shape_dir, filename + '.txt'),
                         shape_dir_to_shape_id[shape_dir])
                    )
        self.file_paths = file_paths
        self.num_shapes = 16
        self.num_classes = 50

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            coords, normal, label, shape_id = self.cache[index]
        else:
            file_path, shape_id = self.file_paths[index]
            data = np.loadtxt(file_path).astype(np.float32)
            coords = data[:, :3]
            if self.normalize:
                coords = self.normalize_point_cloud(coords)
            normal = data[:, 3:6]
            label = data[:, -1].astype(np.int64)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (coords, normal, label, shape_id)

        choice = np.random.choice(label.shape[0], self.num_points, replace=True)
        coords = coords[choice, :].transpose()
        if self.jitter:
            coords = self.jitter_point_cloud(coords)
        if self.with_normal:
            normal = normal[choice, :].transpose()
            if self.with_one_hot_shape_id:
                shape_one_hot = np.zeros((self.num_shapes, self.num_points), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, normal, shape_one_hot])
            else:
                point_set = np.concatenate([coords, normal])
        else:
            if self.with_one_hot_shape_id:
                shape_one_hot = np.zeros((self.num_shapes, self.num_points), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, shape_one_hot])
            else:
                point_set = coords

        shape_label = np.array([1])
        shape_label = shape_label + shape_id

        return point_set, label[choice].transpose(), shape_label

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def normalize_point_cloud(points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        return points / np.max(np.linalg.norm(points, axis=1))

    @staticmethod
    def jitter_point_cloud(points, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              3xN array, original batch of point clouds
            Return:
              3xN array, jittered batch of point clouds
        """
        assert (clip > 0)
        return np.clip(sigma * np.random.randn(*points.shape), -1 * clip, clip).astype(np.float32) + points

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area=5, with_normalized_coords=True):
        """
        :param num_points: number of points to process for each scene
        :param partition: 'train' or 'test'
        :param with_normalized_coords: whether include the normalized coords in features (default: True)
        :param test_area: which area to holdout (default: 5)
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.join(BASE_DIR, 'data', 's3dis', 'pointcnn')
        self.partition = partition
        self.num_points = num_points
        self.test_area = None if test_area is None else int(test_area)
        self.with_normalized_coords = with_normalized_coords
        # keep at most 20/30 files in memory
        self.cache_size = 20 if partition == 'train' else 30
        self.cache = {}

        # mapping batch index to corresponding file
        areas = []
        if self.partition == 'train':
            for a in range(1, 7):
                if a != self.test_area:
                    areas.append(os.path.join(self.root, f'Area_{a}'))
        else:
            areas.append(os.path.join(self.root, f'Area_{self.test_area}'))

        self.num_scene_windows, self.max_num_points = 0, 0
        index_to_filename, scene_list = [], {}
        filename_to_start_index = {}
        for area in areas:
            area_scenes = os.listdir(area)
            area_scenes.sort()
            for scene in area_scenes:
                current_scene = os.path.join(area, scene)
                scene_list[current_scene] = []
                for partition in ['zero', 'half']:
                    current_file = os.path.join(current_scene, f'{partition}_0.h5')
                    filename_to_start_index[current_file] = self.num_scene_windows
                    h5f = h5py.File(current_file, 'r')
                    num_windows = h5f['data'].shape[0]
                    self.num_scene_windows += num_windows
                    for i in range(num_windows):
                        index_to_filename.append(current_file)
                    scene_list[current_scene].append(current_file)
        self.index_to_filename = index_to_filename
        self.filename_to_start_index = filename_to_start_index
        self.scene_list = scene_list

    def __len__(self):
        return self.num_scene_windows

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        if filename not in self.cache.keys():
            h5f = h5py.File(filename, 'r')
            scene_data = h5f['data']
            scene_label = h5f['label_seg']
            scene_num_points = h5f['data_num']
            if len(self.cache.keys()) < self.cache_size:
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
            else:
                victim_idx = np.random.randint(0, self.cache_size)
                cache_keys = list(self.cache.keys())
                cache_keys.sort()
                self.cache.pop(cache_keys[victim_idx])
                self.cache[filename] = (scene_data, scene_label, scene_num_points)
        else:
            scene_data, scene_label, scene_num_points = self.cache[filename]

        internal_pos = index - self.filename_to_start_index[filename]
        current_window_data = np.array(scene_data[internal_pos]).astype(np.float32)
        current_window_label = np.array(scene_label[internal_pos]).astype(np.int64)
        current_window_num_points = scene_num_points[internal_pos]

        choices = np.random.choice(current_window_num_points, self.num_points,
                                   replace=(current_window_num_points < self.num_points))
        data = current_window_data[choices, ...].transpose()
        label = current_window_label[choices]
        # data[9, num_points] = [x_in_block, y_in_block, z_in_block, r, g, b, x / x_room, y / y_room, z / z_room]
        if self.with_normalized_coords:
            return data, label
        else:
            return data[:-3, :], label

# ──────────────────────────────────────────────────────────────────────────────
# ─── NEW: ScanObjectNNDataset definition ─────────────────────────────────────

class ScanObjectNNDatasetModified(Dataset):
    """
    Loads ScanObjectNN from H5 files under either:
      • data/ScanObjectNN/train/training_objectdataset_augmentedrot_scale75.h5
        data/ScanObjectNN/test/test_objectdataset_augmentedrot_scale75.h5
    or, if --dev_scan_subset is passed:
      • data/dev_scanObjectNN_subset/train/training_subset_10.h5
        data/dev_scanObjectNN_subset/test/test_subset_10.h5
    Returns a (npoint × 6) array per sample: (X,Y,Z,Nx,Ny,Nz).
    """
    def __init__(self, npoint=1024, partition='train', args=None, knn_normals=30):
        super().__init__()
        self.npoint = npoint
        self.partition = partition  # 'train' or 'test'
        self.args = args
        self.knn_normals = knn_normals

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # 1) Decide which “root folder” to use
        if getattr(args, 'dev_scan_subset', False):
            DATA_DIR = os.path.join(BASE_DIR, 'data', 'dev_scanObjectNN_subset')
        else:
            DATA_DIR = os.path.join(BASE_DIR, 'data', 'ScanObjectNN')

        # 2) Pick the correct H5 filename
        if getattr(args, 'dev_scan_subset', False):
            # for the 10‐sample dev subset
            if partition == 'train':
                h5_name = 'training_subset_10.h5'
            else:
                h5_name = 'test_subset_10.h5'
        else:
            # for the full ScanObjectNN
            if partition == 'train':
                h5_name = 'training_objectdataset_augmentedrot_scale75.h5'
            else:
                h5_name = 'test_objectdataset_augmentedrot_scale75.h5'

        # 3) Construct full path (DATA_DIR/<train-or-test>/<h5_name>)
        h5_path = os.path.join(DATA_DIR, partition, h5_name)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Cannot find {h5_path}. "
                                    f"Make sure the H5 file is in the correct {{train,test}} folder.")

        # 4) Load entire H5 into memory
        with h5py.File(h5_path, 'r') as f:
            self.raw_data = f['data'][:]    # shape = (N_samples, M_pts, 3)
            self.raw_label = f['label'][:]  # shape = (N_samples,)

        # 5) Precompute normals for every point cloud (one pass)
        self.normals_list = [None] * self.raw_data.shape[0]
        for idx in range(self.raw_data.shape[0]):
            pts = self.raw_data[idx]  # (M, 3)
            self.normals_list[idx] = self._compute_normals(pts)  # returns (M, 3)

        # 6) Load the shape‐names file into a list of strings
        #    scanobjectnn_shape_names.txt should list one class name per line, in label order.
        names_path = os.path.join(DATA_DIR, 'scanobjectnn_shape_names.txt')
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Cannot find {names_path}. "
                                    f"Make sure scanobjectnn_shape_names.txt is placed under {DATA_DIR}")
        self.shape_names = []
        with open(names_path, 'r') as f:
            for line in f:
                nm = line.strip()
                if nm:
                    self.shape_names.append(nm)
        # Sanity check: there should be exactly 20 names
        if len(self.shape_names) != 15:
            raise ValueError(f"Expected 15 class names, but found {len(self.shape_names)} in {names_path}")

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
          sampled_feats: (npoint, 6) float32 array = (X,Y,Z,Nx,Ny,Nz)
          label:         int (0..num_classes-1)
        """
        pts = self.raw_data[idx]           # (M, 3)
        nrm = self.normals_list[idx]       # (M, 3)
        label = int(self.raw_label[idx])   # scalar

        M = pts.shape[0]
        # 1) Randomly sample exactly npoint indices (duplicate if M < npoint)
        if M >= self.npoint:
            choice = np.random.choice(M, self.npoint, replace=False)
        else:
            choice = np.random.choice(M, self.npoint, replace=True)

        sampled_pts = pts[choice, :]   # (npoint, 3)
        sampled_nrm = nrm[choice, :]   # (npoint, 3)

        # 2) Normalize XYZ to unit sphere
        sampled_pts[:, 0:3] = pc_normalize(sampled_pts[:, 0:3])

        # 3) Concatenate to (npoint, 6)
        sampled_feats = np.concatenate([sampled_pts, sampled_nrm], axis=1).astype('float32')

        # 4) Look up the class name by integer label
        classname = self.shape_names[label]

        return sampled_feats, label, classname

    def _compute_normals(self, pts_np):
        """
        Estimate per-point normals via PCA over knn_normals neighbors.
        Input:
          pts_np: (M,3) NumPy array
        Returns:
          normals: (M,3) NumPy array, unit-length normals
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.knn_normals),
            fast_normal_computation=False
        )
        normals = np.asarray(pcd.normals, dtype='float32')
        return normals

class ScanObjectNNDataset(Dataset):
    """
    Loads ScanObjectNN from H5 files under either:
      • data/ScanObjectNN/train/training_objectdataset_augmentedrot_scale75.h5
        data/ScanObjectNN/test/test_objectdataset_augmentedrot_scale75.h5
    or, if --dev_scan_subset is passed:
      • data/dev_scanObjectNN_subset/train/training_subset_10.h5
        data/dev_scanObjectNN_subset/test/test_subset_10.h5
    Returns a (npoint × 6) array per sample: (X,Y,Z,Nx,Ny,Nz).
    """
    def __init__(self, npoint=1024, partition='train', args=None, knn_normals=30):
        super().__init__()
        self.npoint = npoint
        self.partition = partition  # 'train' or 'test'
        self.args = args
        self.knn_normals = knn_normals

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # 1) Decide which “root folder” to use
        if getattr(args, 'dev_scan_subset', False):
            DATA_DIR = os.path.join(BASE_DIR, 'data', 'dev_scanObjectNN_subset')
        else:
            DATA_DIR = os.path.join(BASE_DIR, 'data', 'ScanObjectNN')

        # 2) Pick the correct H5 filename
        if getattr(args, 'dev_scan_subset', False):
            # for the 10‐sample dev subset
            if partition == 'train':
                h5_name = 'training_subset_10.h5'
            else:
                h5_name = 'test_subset_10.h5'
        else:
            # for the full ScanObjectNN
            if partition == 'train':
                h5_name = 'training_objectdataset_augmentedrot_scale75.h5'
            else:
                h5_name = 'test_objectdataset_augmentedrot_scale75.h5'

        # 3) Construct full path (DATA_DIR/<train-or-test>/<h5_name>)
        h5_path = os.path.join(DATA_DIR, partition, h5_name)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Cannot find {h5_path}. "
                                    f"Make sure the H5 file is in the correct {{train,test}} folder.")

        # 4) Load entire H5 into memory
        with h5py.File(h5_path, 'r') as f:
            self.raw_data = f['data'][:]    # shape = (N_samples, M_pts, 3)
            self.raw_label = f['label'][:]  # shape = (N_samples,)

        # 5) Precompute normals for every point cloud (one pass)
        self.normals_list = [None] * self.raw_data.shape[0]
        for idx in range(self.raw_data.shape[0]):
            pts = self.raw_data[idx]  # (M, 3)
            self.normals_list[idx] = self._compute_normals(pts)  # returns (M, 3)


    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
          sampled_feats: (npoint, 6) float32 array = (X,Y,Z,Nx,Ny,Nz)
          label:         int (0..num_classes-1)
        """
        pts = self.raw_data[idx]           # (M, 3)
        nrm = self.normals_list[idx]       # (M, 3)
        label = int(self.raw_label[idx])   # scalar

        M = pts.shape[0]
        # 1) Randomly sample exactly npoint indices (duplicate if M < npoint)
        if M >= self.npoint:
            choice = np.random.choice(M, self.npoint, replace=False)
        else:
            choice = np.random.choice(M, self.npoint, replace=True)

        sampled_pts = pts[choice, :]   # (npoint, 3)
        sampled_nrm = nrm[choice, :]   # (npoint, 3)

        # 2) Normalize XYZ to unit sphere
        sampled_pts[:, 0:3] = pc_normalize(sampled_pts[:, 0:3])

        # 3) Concatenate to (npoint, 6)
        sampled_feats = np.concatenate([sampled_pts, sampled_nrm], axis=1).astype('float32')

        return sampled_feats, label

    def _compute_normals(self, pts_np):
        """
        Estimate per-point normals via PCA over knn_normals neighbors.
        Input:
          pts_np: (M,3) NumPy array
        Returns:
          normals: (M,3) NumPy array, unit-length normals
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.knn_normals),
            fast_normal_computation=False
        )
        normals = np.asarray(pcd.normals, dtype='float32')
        return normals

