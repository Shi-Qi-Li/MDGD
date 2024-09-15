from typing import List, Dict, Optional

import os
import pickle
import torch
import numpy as np
import numpy.typing as npt
import open3d as o3d
from torch.utils.data import Dataset

from utils import square_distance, random_rotation_matrix, inv_transform, pairwise_registration

from .builder import DATASET

@DATASET
class Scene(Dataset):
    def __init__(self, data_path:str, split:str, ird: float, point_sample: bool, frame_sample: bool, frame_limit: Optional[List] = None, point_limit: Optional[int] = 5000, overlap_only: bool = False, processes: int = 8):
        super().__init__()
        assert split in ["train", "val", "3dmatch", "eth", "scannet"], "Split must be one of [train, val, 3dmatch, eth, scannet]"
        self.data_path = data_path
        self.split = split
        self.ird = ird

        self.point_sample = point_sample
        self.frame_sample = frame_sample
        self.overlap_only = overlap_only

        self.frame_limit = frame_limit
        self.point_limit = point_limit
        self.pair_limit = 16
        self.node_limit = 1000

        self.use_mutual = False

        self.rot_range = 180
        self.trans_range = 6
        self.noise_range = 0.01

        self.processes = processes

        self._load_metadata()

    def __len__(self):
        return len(self.metadata)

    def _load_gt(self, scene: str) -> npt.NDArray:
        gt_path = os.path.join(self.data_path, self.split, scene, "PointCloud", "gt.log")
        with open(gt_path,"r") as f:
            lines = f.readlines()
            pair_num = len(lines) // 5
            pairs_id = []
            for k in range(pair_num):
                id0, id1 = np.fromstring(lines[k * 5], dtype=np.float32, sep='\t')[0:2]
                id0 = int(id0)
                id1 = int(id1)
                pairs_id.append([id0, id1])

        pairs_id = np.array(pairs_id)

        return pairs_id
    
    def _load_pc(self, scene: str, index: int) -> npt.NDArray:
        pc_path = os.path.join(self.data_path, self.split, scene, "PointCloud", f"cloud_bin_{index}.ply")
        pc = o3d.io.read_point_cloud(pc_path)
        pc = np.asarray(pc.points).astype(np.float32)
        return pc

    def _load_keypoints_index(self, scene: str, index: int) -> npt.NDArray:
        key_index_path = os.path.join(self.data_path, self.split, scene, "Keypoints", f"cloud_bin_{index}Keypoints.txt")
        keypoints_index = np.loadtxt(key_index_path).astype(np.int64)
        return keypoints_index

    def _load_keypoints(self, scene: str, index: int) -> npt.NDArray:
        keypoints_path = os.path.join(self.data_path, self.split, scene, "Keypoints_PC", f"cloud_bin_{index}Keypoints.npy")
        if not os.path.exists(keypoints_path):
            pc = self._load_pc(scene, index)
            keypoints_index = self._load_keypoints_index(scene, index)
            keypoints = pc[keypoints_index]
            if not os.path.exists(os.path.join(self.data_path, self.split, scene, "Keypoints_PC")):
                os.makedirs(os.path.join(self.data_path, self.split, scene, "Keypoints_PC"))
            np.save(keypoints_path, keypoints)
        else:
            keypoints = np.load(keypoints_path).astype(np.float32)
        return keypoints
    
    def _load_descriptor(self, scene: str, index: int) -> npt.NDArray:
        descriptor_path = os.path.join(self.data_path, self.split, scene, "yoho_desc", f"{index}.npy")
        descriptor = np.load(descriptor_path).astype(np.float32)
        return descriptor

    def _load_metadata(self):
        self.metadata = []
        pkl_path = os.path.join(self.data_path, f"{self.split}.pkl")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            for scene in data:
                name, overlap = scene[0], scene[1]
                self.metadata.append({
                    "scene": name,
                    "frames": overlap.shape[0],
                    "overlap": overlap
                })

    def _sample_frame(self, gt_overlap: npt.NDArray):
        frames = gt_overlap.shape[0]

        if self.frame_sample == False or frames <= self.frame_limit[0]:
            return np.arange(frames), gt_overlap
        
        sample_num = np.random.choice(np.arange(self.frame_limit[0],self.frame_limit[1]), 1)[0]
        frame_index = np.random.permutation(np.arange(frames))[0:sample_num]

        gt_overlap = gt_overlap[frame_index,:]
        gt_overlap = gt_overlap[:,frame_index]
        return frame_index, gt_overlap

    def _sample_points(self, keypoints: npt.NDArray, descriptors: npt.NDArray):
        if self.point_sample == False: 
            return keypoints, descriptors
        # Determine how many points to be sampled
        k = np.random.choice(np.arange(1024, self.point_limit), 1)[0]
        # random center
        center = np.random.choice(keypoints.shape[0], 1)[0]
        center = keypoints[center]
        # calculate Knn of the selected center
        cpdist = np.sum(np.square(keypoints - center[None,:]), axis=-1)
        argp = np.argsort(cpdist)[0:int(1.2 * k)]
        # resample n_sample points
        index = np.random.permutation(argp)[0:k]
        # the down sampled keypoints
        keypoints_sample = keypoints[index]
        descriptors_sample = descriptors[index]
        return keypoints_sample, descriptors_sample    
    
    def _calculate_overlap(self, keypoints: npt.NDArray, gt_overlap: npt.NDArray, ird: float = 0.08):
        if self.point_sample == False: 
            return gt_overlap
        
        frames = len(keypoints)
        overlap = np.zeros([frames, frames], dtype=np.float32)
        for i in range(frames):
            for j in range(i + 1, frames):
                if gt_overlap[i, j] == 0:
                    overlap[i, j], overlap[j, i] = 0, 0
                    continue
                pc_i, pc_j = keypoints[i], keypoints[j]
                dist = square_distance(pc_i, pc_j).numpy()
                # determine the minimum distance
                mi = np.min(dist, axis=1)   
                mj = np.min(dist, axis=0)              
                overlap_ij = np.sum(mi < ird * ird) + np.sum(mj < ird * ird)       
                overlap_ij /= (pc_i.shape[0] + pc_j.shape[0])
                overlap[i,j], overlap[j,i] = overlap_ij, overlap_ij
        return overlap

    def _aug_point_clouds(self, keypoints_list: List[npt.NDArray], keypoints_all: List[npt.NDArray]):
        aug_Ts = np.eye(4, dtype=np.float32)[None].repeat(len(keypoints_list),axis=0)
        if self.split != "train" and self.split!= "val": 
            return keypoints_list, keypoints_all, aug_Ts
        # random rotation
        else:
            aug_keypoints_list = []
            aug_keypoints_all = []
            for i, (points, points_all) in enumerate(zip(keypoints_list, keypoints_all)):
                aug_r = random_rotation_matrix(self.rot_range)
                aug_t = (np.random.rand(1,3) - 0.5) * self.trans_range
                aug_n = (np.random.rand(points.shape[0], 3) - 0.5) * self.noise_range
                aug_r, aug_t, aug_n = aug_r.astype(np.float32), aug_t.astype(np.float32), aug_n.astype(np.float32)
                # apply to the point cloud
                points = points @ aug_r.T + aug_t
                points_all = points_all @ aug_r.T + aug_t

                aug_keypoints_list.append(points)
                aug_keypoints_all.append(points_all)
                # save the augmentation transformation
                aug_Ts[i, 0:3, 0:3], aug_Ts[i, 0:3, 3] = aug_r, aug_t
        return aug_keypoints_list, aug_keypoints_all, aug_Ts
    
    def _calculate_pairwise(self, keypoints_list: List[npt.NDArray], descriptors_list: List[npt.NDArray]):
        frames = len(keypoints_list)

        rel_mat = np.zeros((frames, frames, 4, 4), dtype=np.float32)
        ir_mat = np.zeros((frames, frames, 2), dtype=np.float32)
        match_mat = np.zeros((frames, frames, 8), dtype=np.float32)

        pool = torch.multiprocessing.Pool(processes=self.processes)
    
        tasks = [(keypoints_list[i], keypoints_list[j], descriptors_list[i], descriptors_list[j], self.ird) for i in range(frames) for j in range(i + 1, frames)]
        task_id = [(i, j) for i in range(frames) for j in range(i + 1, frames)]
        results_list = pool.starmap_async(pairwise_registration, tasks)
        results_list = results_list.get()
    
        for (i, j), (trans_est, ir_info, match_info) in zip(task_id, results_list):
            rel_mat[i, j] = trans_est
            rel_mat[j, i] = inv_transform(trans_est)
            ir_mat[i, j] = ir_mat[j, i] = ir_info
            match_mat[i, j] = match_mat[j, i] = match_info

        pool.close()
        pool.join()

        for i in range(frames):
            rel_mat[i, i] = np.eye(4, dtype=np.float32)
            
        return rel_mat, ir_mat, match_mat
    
    def _create_relative_gt(self, abs_gt: npt.NDArray):
        rel_gt = abs_gt.reshape(-1, 4) @ inv_transform(abs_gt).transpose(1, 0, 2).reshape(4, -1)

        return rel_gt

    def _sample_pairs(self, gt_overlap: npt.NDArray, aug_Ts: npt.NDArray, keypoints_list: List[npt.NDArray], descriptors_list: List[npt.NDArray], ird: float = 0.08):
        frames = gt_overlap.shape[0]
        valid_pairs = []
        for i in range(frames):
            for j in range(i + 1, frames):
                if gt_overlap[i, j] > 0.1:
                    valid_pairs.append((i, j))

        if len(valid_pairs) > self.pair_limit:
            valid_pairs = np.random.permutation(valid_pairs)[0: self.pair_limit]

        corrs, labels, corr_src, corr_tgt, transforms = [], [], [], [], []
        for pair in valid_pairs:
            i, j = pair
            src_sel_ind = np.random.choice(keypoints_list[i].shape[0], self.node_limit)
            tgt_sel_ind = np.random.choice(keypoints_list[j].shape[0], self.node_limit)

            src_desc = np.copy(descriptors_list[i][src_sel_ind, :])
            tgt_desc = np.copy(descriptors_list[j][tgt_sel_ind, :])
            src_keypts = np.copy(keypoints_list[i][src_sel_ind, :])
            tgt_keypts = np.copy(keypoints_list[j][tgt_sel_ind, :])

            # construct the correspondence set by mutual nn in feature space.
            distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
            source_idx = np.argmin(distance, axis=1)
            if self.use_mutual:
                target_idx = np.argmin(distance, axis=0)
                mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
                corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
            else:
                corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)

            # build the ground truth label
            frag1 = src_keypts[corr[:, 0]]
            frag2 = tgt_keypts[corr[:, 1]]
            
            frag1_warp = (frag1 - aug_Ts[i, :3, 3]) @ aug_Ts[i, :3, :3]
            frag1_warp = frag1_warp @ aug_Ts[j, :3, :3].T + aug_Ts[j, :3, 3]

            gt_transform = aug_Ts[j] @ inv_transform(aug_Ts[i])
            
            distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
            label = (distance < ird).astype(np.int32)
            
            # prepare input to the network
            input_src_keypts = src_keypts[corr[:, 0]]
            input_tgt_keypts = tgt_keypts[corr[:, 1]]

            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            corr_pos = corr_pos - corr_pos.mean(axis=0)
            
            corrs.append(corr_pos)
            labels.append(label)
            corr_src.append(input_src_keypts)
            corr_tgt.append(input_tgt_keypts)
            transforms.append(gt_transform)
        
        corrs = np.stack(corrs)
        labels = np.stack(labels)
        corr_src = np.stack(corr_src)
        corr_tgt = np.stack(corr_tgt)
        transforms = np.stack(transforms)

        return corrs, labels, corr_src, corr_tgt, transforms

    def __getitem__(self, index):
    
        scene = self.metadata[index]["scene"]
        gt_overlap = self.metadata[index]["overlap"]

        frame_index, gt_overlap = self._sample_frame(gt_overlap)

        keypoints_list, descriptors_list = [], []
        keypoints_all, descriptors_all = [], []
        for index in frame_index:
            keypoints = self._load_keypoints(scene, index)
            descriptors = self._load_descriptor(scene, index)
            keypoints_all.append(keypoints)
            descriptors_all.append(descriptors)
            keypoints, descriptors = self._sample_points(keypoints, descriptors)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
        
        gt_overlap = self._calculate_overlap(keypoints_list, gt_overlap)
        
        if self.overlap_only:
            data_batch = {
                "descriptors": descriptors_list,
                "gt_overlap": gt_overlap,
            }

            return data_batch
            
        keypoints_list, keypoints_all, abs_gt = self._aug_point_clouds(keypoints_list, keypoints_all)

        rel_mat, ir_mat, match_mat = self._calculate_pairwise(keypoints_all, descriptors_all)
        
        rel_gt = self._create_relative_gt(abs_gt)

        data_batch = {
            "keypoints_all": keypoints_all,
            "descriptors": descriptors_list,
            "gt_overlap": gt_overlap,
            "rel_mat": rel_mat,
            "ir_mat": ir_mat,
            "match_mat": match_mat,
            "abs_gt": abs_gt,
            "rel_gt": rel_gt,
            "scene": scene
        }

        if self.split != "train":
            data_batch["pairs_id"] = self._load_gt(scene)
            data_batch["gt_file_path"] = os.path.join(self.data_path, self.split, scene, "PointCloud")

        return data_batch

@DATASET
class Cache(Dataset):
    def __init__(self, data_path: str, cache_folder: str, total_epoch: int, split: str = "train"):
        super().__init__()
        self.data_path = data_path
        self.cache_folder = cache_folder
        
        self.split = split
        if split == "train":
            scene_num = 48

        self.data_list = [f"{epoch}_{index}" for index in range(scene_num) for epoch in range(total_epoch)]

    def __len__(self):
        return len(self.data_list)

    def _load_gt(self, scene: str) -> npt.NDArray:
        gt_path = os.path.join(self.data_path, self.split, scene, "PointCloud", "gt.log")
        with open(gt_path,"r") as f:
            lines = f.readlines()
            pair_num = len(lines) // 5
            pairs_id = []
            for k in range(pair_num):
                id0, id1 = np.fromstring(lines[k * 5], dtype=np.float32, sep='\t')[0:2]
                id0 = int(id0)
                id1 = int(id1)
                pairs_id.append([id0, id1])

        pairs_id = np.array(pairs_id)

        return pairs_id
    
    def _load_pc(self, scene: str, index: int) -> npt.NDArray:
        pc_path = os.path.join(self.data_path, self.split, scene, "PointCloud", f"cloud_bin_{index}.ply")
        pc = o3d.io.read_point_cloud(pc_path)
        pc = np.asarray(pc.points).astype(np.float32)
        return pc

    def _load_keypoints_index(self, scene: str, index: int) -> npt.NDArray:
        key_index_path = os.path.join(self.data_path, self.split, scene, "Keypoints", f"cloud_bin_{index}Keypoints.txt")
        keypoints_index = np.loadtxt(key_index_path).astype(np.int64)
        return keypoints_index

    def _load_keypoints(self, scene: str, index: int) -> npt.NDArray:
        keypoints_path = os.path.join(self.data_path, self.split, scene, "Keypoints_PC", f"cloud_bin_{index}Keypoints.npy")
        if not os.path.exists(keypoints_path):
            pc = self._load_pc(scene, index)
            keypoints_index = self._load_keypoints_index(scene, index)
            keypoints = pc[keypoints_index]
            if not os.path.exists(os.path.join(self.data_path, self.split, scene, "Keypoints_PC")):
                os.makedirs(os.path.join(self.data_path, self.split, scene, "Keypoints_PC"))
            np.save(keypoints_path, keypoints)
        else:
            keypoints = np.load(keypoints_path).astype(np.float32)
        return keypoints

    def _load_descriptor(self, scene: str, index: int) -> npt.NDArray:
        descriptor_path = os.path.join(self.data_path, self.split, scene, "yoho_desc", f"{index}.npy")
        descriptor = np.load(descriptor_path).astype(np.float32)
        return descriptor
    
    def __getitem__(self, index):
        pickle_path = os.path.join(self.data_path, self.cache_folder, f"{self.data_list[index]}.pkl")

        with open(pickle_path, 'rb') as f:
            data_batch = pickle.load(f)
        
        if self.split != "train":
            data_batch["pairs_id"] = self._load_gt(data_batch["scene"])
            data_batch["gt_file_path"] = os.path.join(self.data_path, self.split, data_batch["scene"], "PointCloud")
        
        if "descriptors" not in data_batch:
            frame_num = data_batch["gt_overlap"].shape[0]
            descriptors = [self._load_descriptor(data_batch["scene"], index) for index in range(frame_num)]
            data_batch["descriptors"] = descriptors
            keypoints = [self._load_keypoints(data_batch["scene"], index) for index in range(frame_num)]
            data_batch["keypoints_all"] = keypoints
            del data_batch["scene"]
        
        return data_batch