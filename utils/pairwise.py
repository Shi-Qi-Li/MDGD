from typing import Union, Tuple

import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import open3d as o3d

import knn_search

from utils import transform, integrate_trans

def nearest_search(ref: Union[torch.Tensor, npt.NDArray], query: Union[torch.Tensor, npt.NDArray]):
    """
    :param ref: (dim, num)
    :param query: (dim, num)    
    """

    d, i = knn_search.knn_search(ref, query, 1)
    i -= 1
    return d, i

def mutual_match(descriptor0, descriptor1):
    """
    descriptor0: (num, dim)
    descriptor1: (num, dim)
    """

    if isinstance(descriptor0, np.ndarray):
        descriptor0 = torch.from_numpy(descriptor0).float()
    if isinstance(descriptor1, np.ndarray):
        descriptor1 = torch.from_numpy(descriptor1).float()

    descriptor0, descriptor1 = descriptor0.T.contiguous().cuda(), descriptor1.T.contiguous().cuda()

    d0, i0 = nearest_search(descriptor1, descriptor0)
    d1, i1 = nearest_search(descriptor0, descriptor1)

    i0 = i0[0].cpu().numpy()
    i1 = i1[0].cpu().numpy()

    d0 = d0[0].cpu().numpy()

    match_idx = np.where(i1[i0] == np.arange(len(i0)))[0]

    match = np.stack([np.arange(len(i0)), i0], axis=-1)
    match = match[match_idx]
    dis = d0[match_idx]

    return match, dis

def ransac(pc0: npt.NDArray, pc1: npt.NDArray, match: npt.NDArray, iteration: int = 50000, max_correspondence_distance: float = 0.07):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(pc0)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(pc1)
    coores = o3d.utility.Vector2iVector(match)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_pcd, target_pcd, coores, max_correspondence_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        o3d.pipelines.registration.RANSACConvergenceCriteria(iteration, 1000)
    )

    trans = result.transformation
    trans = np.linalg.inv(trans)

    return trans

def svd(pc0: npt.NDArray, pc1: npt.NDArray, scores: npt.NDArray):
    scores = scores / np.sum(scores)

    centroid0 = np.sum(pc0 * scores[:, None], axis=0, keepdims=True)
    centroid1 = np.sum(pc1 * scores[:, None], axis=0, keepdims=True)

    pc0_centered = pc0 - centroid0
    pc1_centered = pc1 - centroid1

    weight=np.diag(scores)
    H = np.matmul(np.matmul(np.transpose(pc0_centered), weight), pc1_centered)
    U, _, V = np.linalg.svd(H)
    R = np.matmul(U, V)

    if np.linalg.det(R) < 0:
        R[0:2] = R[[1,0]]

    t = centroid0 - centroid1 @ R.T

    trans = integrate_trans(R, t.reshape(3, 1))

    return trans

def inlier_ratio(kpc0, kpc1, trans, max_correspondence_distance: float = 0.07):
    kpc1_t = transform(kpc1, trans)
    dis = np.sum(np.square(kpc0 - kpc1_t), axis=-1)
    inlier_index = np.where(dis < max_correspondence_distance ** 2)[0]
    ir = inlier_index.shape[0] / dis.shape[0]

    if inlier_index.shape[0] < 2:
        pr = 1
    else:
        overlap_pc = o3d.geometry.PointCloud()
        overlap_pc.points = o3d.utility.Vector3dVector(np.concatenate([kpc0[inlier_index], kpc1_t[inlier_index]], axis=0))
        plane_model, plane_inliers = overlap_pc.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        pr = len(plane_inliers) / (inlier_index.shape[0] * 2)

    inlier_info = np.array([ir, pr], dtype=np.float32)

    return inlier_info

def refine(kpc0, kpc1, trans, scores, max_correspondence_distance: float = 0.07):
    kpc1_t = transform(kpc1, trans)
    diff = np.sum(np.square(kpc0 - kpc1_t), axis=-1)
    overlap = np.where(diff < max_correspondence_distance ** 2)[0]

    kpc0 = kpc0[overlap]
    kpc1 = kpc1[overlap]
    scores = scores[overlap]

    trans = svd(kpc0, kpc1, scores)

    return trans

def pairwise_registration(
        pc0: npt.NDArray,
        pc1: npt.NDArray,
        descriptor0: npt.NDArray,
        descriptor1: npt.NDArray,
        max_correspondence_distance: float = 0.07
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

    match, dis = mutual_match(descriptor0, descriptor1)
    trans = ransac(pc0, pc1, match)

    kpc0 = pc0[match[:, 0]]
    kpc1 = pc1[match[:, 1]]

    scores = np.ones(match.shape[0])

    trans = refine(kpc0, kpc1, trans, scores, max_correspondence_distance * 2)
    trans = refine(kpc0, kpc1, trans, scores, max_correspondence_distance)

    inlier_info = inlier_ratio(kpc0, kpc1, trans, max_correspondence_distance)

    ecdf30 = np.sum(dis < 0.30) / match.shape[0]
    ecdf35 = np.sum(dis < 0.35) / match.shape[0]
    ecdf40 = np.sum(dis < 0.40) / match.shape[0]
    ecdf45 = np.sum(dis < 0.45) / match.shape[0]
    mean = np.mean(dis)
    median = np.median(dis)
    std = np.std(dis)

    match_info = np.array([match.shape[0] / descriptor0.shape[0], ecdf30, ecdf35, ecdf40, ecdf45, mean, median, std], dtype=np.float32)

    return trans, inlier_info, match_info