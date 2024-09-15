from typing import Dict

import os
import torch
import numpy as np
import nibabel as nib

from utils import integrate_trans, inv_transform

def pairwise_metrics(predictions: Dict, ground_truth: Dict) -> Dict:
    te_thre = 30
    re_thre = 15
    
    transform_gt = ground_truth["transform_gt"]
    transform_pred = predictions["transform_pred"].detach()

    pairs = transform_gt.shape[0]
    truely_reg = 0

    R_pred = transform_pred[:, :3, :3]
    t_pred = transform_pred[:, :3, 3]

    R_gt = transform_gt[:, :3, :3]
    t_gt = transform_gt[:, :3, 3]

    for p in range(pairs):
        re = torch.acos(torch.clamp((torch.trace(R_pred[p].T @ R_gt[p]) - 1) / 2.0, min=-1, max=1))
        te = torch.sqrt(torch.sum((t_pred[p] - t_gt[p]) ** 2))

        re = re * 180 / np.pi
        te = te * 100
        if te < te_thre and re < re_thre:
            truely_reg += 1

    RR = np.array(truely_reg / pairs)
    
    metrics = {
        "RR": RR
    }
    
    return metrics

def read_trajectory(filename, dim: int = 4):
    """
    Function that reads a trajectory saved in the 3DMatch/Redwood format to a numpy array. 
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html
    
    Args:
    filename (str): path to the '.txt' file containing the trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)
    Returns:
    final_keys (dict): indices of pairs with more than 30% overlap (only this ones are included in the gt file)
    traj (numpy array): gt pairwise transformation matrices for n pairs[n,dim,dim] 
    """

    with open(filename) as f:
        lines = f.readlines()

        # Extract the point cloud pairs
        keys = lines[0::(dim + 1)]
        temp_keys = []
        for i in range(len(keys)):
            temp_keys.append(keys[i].split('\t')[0:3])

        final_keys = []
        for i in range(len(temp_keys)):
            final_keys.append([temp_keys[i][0].strip(), temp_keys[i][1].strip(), temp_keys[i][2].strip()])

        traj = []
        for i in range(len(lines)):
            if i % 5 != 0:
                traj.append(lines[i].split('\t')[0:dim])

        traj = np.asarray(traj, dtype=np.float32).reshape(-1, dim, dim)
        
        final_keys = np.asarray(final_keys, dtype=np.int64)

        return final_keys, traj

def read_trajectory_info(filename, dim: int = 6):
    """
    Function that reads the trajectory information saved in the 3DMatch/Redwood format to a numpy array.
    Information file contains the variance-covariance matrix of the transformation paramaters. 
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html
    
    Args:
    filename (str): path to the '.txt' file containing the trajectory information data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)
    Returns:
    n_frame (int): number of fragments in the scene
    cov_matrix (numpy array): covariance matrix of the transformation matrices for n pairs[n,dim,dim] 
    """

    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    assert (len(contents) == 7 * n_pairs)
    info_list = []
    n_frame = 0

    for i in range(n_pairs):
        frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
        info_matrix = np.concatenate(
            [np.fromstring(item, sep='\t').reshape(1, -1) for item in contents[i * 7 + 1:i * 7 + 7]], axis=0)
        info_list.append(info_matrix)
    
    cov_matrix = np.asarray(info_list, dtype=np.float32).reshape(-1, dim, dim)
    
    return n_frame, cov_matrix


def computeTransformationErr(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html
    
    Args:
    trans (numpy array): transformation matrices [n,4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [n,6,6]
    Returns:
    p (float): transformation error
    """
    
    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nib.quaternions.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / (info[0, 0] + 1e-6)
    
    return p.item()

def evaluate_registration_RMSE(num_fragment, result, result_pairs, gt_pairs, gt, gt_info, err2=0.2, nonconsecutive=True):
    """
    Evaluates the performance of the registration algorithm according to the evaluation protocol defined
    by the 3DMatch/Redwood datasets. The evaluation protocol can be found at http://redwood-data.org/indoor/registration.html
    
    Args:
    num_fragment (int): number of fragments in the scene
    result (numpy array): estimated transformation matrices [n,4,4]
    result_pairs (numpy array): indices of the point cloud for which the transformation matrix was estimated (m,3)
    gt_pairs (numpy array): indices of the ground truth overlapping point cloud pairs (n,3)
    gt (numpy array): ground truth transformation matrices [n,4,4]
    gt_cov (numpy array): covariance matrix of the ground truth transfromation parameters [n,6,6]
    err2 (float): threshold for the RMSE of the gt correspondences (default: 0.2m)
    Returns:
    precision (float): mean registration precision over the scene (not so important because it can be increased see papers)
    recall (float): mean registration recall over the scene (deciding parameter for the performance of the algorithm)
    """

    err2 = err2 ** 2
    gt_mask = np.zeros((num_fragment, num_fragment), dtype=np.int64)
    flags=[]
    errors=[]

    if nonconsecutive:
        for idx in range(gt_pairs.shape[0]):
            i = int(gt_pairs[idx,0])
            j = int(gt_pairs[idx,1])

            # Only non consecutive pairs are tested
            if abs(j - i) > 1:
                gt_mask[i, j] = idx

        n_gt = np.sum(gt_mask > 0)
    else:
        for idx in range(gt_pairs.shape[0]):
            i = int(gt_pairs[idx,0])
            j = int(gt_pairs[idx,1])
            gt_mask[i, j] = idx
        n_gt = np.sum(gt_mask > 0) + 1

    good = 0
    n_res = 0
    if not nonconsecutive:
        start_check=1
        n_res += 1
        i = int(result_pairs[0,0])
        j = int(result_pairs[0,1])
        pose = result[0,:,:]
        gt_idx = gt_mask[i, j]
        p = computeTransformationErr(np.linalg.inv(gt[0,:,:]) @ pose, gt_info[0,:,:])
        errors.append(np.sqrt(p))
        if p <= err2:
            good += 1
            flags.append(0)
        else:
            flags.append(1)
    else:
        start_check=0

    re, te, rre, rte = [], [], [], []
    for idx in range(start_check,result_pairs.shape[0]):
        i = int(result_pairs[idx,0])
        j = int(result_pairs[idx,1])
        pose = result[idx,:,:]

        if gt_mask[i, j] > 0:
            n_res += 1
            gt_idx = gt_mask[i, j]
            p = computeTransformationErr(np.linalg.inv(gt[gt_idx,:,:]) @ pose, gt_info[gt_idx,:,:])
            e_r = np.rad2deg(np.arccos(np.clip((np.trace(pose[:3, :3].T @ gt[gt_idx, :3, :3]) - 1) / 2, -1, 1)))
            e_t = np.sqrt(np.sum((pose[:3, 3] - gt[gt_idx, :3, 3]) ** 2))
            re.append(e_r)
            te.append(e_t)
            errors.append(np.sqrt(p))
            if p <= err2:
                good += 1
                rre.append(e_r)
                rte.append(e_t)
                flags.append(0)
            else:
                flags.append(1)
        else:
            flags.append(2)
    
    if n_res == 0:
        n_res += 1e6
    
    precision = good * 1.0 / n_res
    recall = good * 1.0 / n_gt

    re = np.mean(np.stack(re))
    te = np.mean(np.stack(te))
    rre = np.mean(np.stack(rre)) if len(rre) > 0 else 0
    rte = np.mean(np.stack(rte)) if len(rte) > 0 else 0

    return precision, recall, flags, errors, re, te, rre, rte

def compute_R_diff(R_gt, R):
    eps = 1e-15
    
    q_gt = nib.quaternions.mat2quat(R_gt)    
    q = nib.quaternions.mat2quat(R)

    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)
    return np.rad2deg(np.abs(err_q))

def evaluate_registration_scannet(result, gt):
    assert result.shape == gt.shape

    num_fragment = result.shape[0]
    re, te = [], []
    for i in range(num_fragment):
        e_r = compute_R_diff(gt[i, :3, :3], result[i, :3, :3])
        e_t = np.sqrt(np.sum((result[i, :3, 3] - gt[i, :3, 3]) ** 2))
        re.append(e_r)
        te.append(e_t)

    re = np.stack(re)
    te = np.stack(te)

    return re, te

def registration_metrics(predictions: Dict, ground_truth: Dict, info: Dict) -> Dict:
    abs_rotation_pred = predictions["abs_rotation_pred"].detach().cpu().numpy()
    abs_translation_pred = predictions["abs_translation_pred"].detach().cpu().numpy()
    abs_pose_pred = integrate_trans(abs_rotation_pred, abs_translation_pred)

    err2 = info["err2"]
    gt_file = os.path.join(info["gt_file_path"][0], "gt.log")
    info_file = os.path.join(info["gt_file_path"][0], "gt.info")

    gt_pairs, gt = read_trajectory(gt_file)
    
    transform_pred = []
    for pair in gt_pairs:
        id0, id1 = pair[0], pair[1]
        trans = abs_pose_pred[id0] @ inv_transform(abs_pose_pred[id1])
        transform_pred.append(trans)
    transform_pred = np.stack(transform_pred, axis=0)

    if info["ecdf"] == False:        
        camera_num, gt_info = read_trajectory_info(info_file)
        precision, recall, flags, errors, re, te, rre, rte = evaluate_registration_RMSE(camera_num, transform_pred, gt_pairs, gt_pairs, gt, gt_info, err2)
        metrics = {
            "RR": recall,
            "RE_mean": re,
            "TE_mean": te,
            "RRE": rre,
            "RTE": rte
        }
    else:
        re, te = evaluate_registration_scannet(transform_pred, gt)
        metrics = {
            "RE_mean": np.mean(re),
            "TE_mean": np.mean(te),
            "RE_median": np.median(re),
            "TE_median": np.median(te)
        }
        for re_threshold in [3, 5, 10, 30, 45]:
            metrics["RE_{}".format(re_threshold)] = np.sum(re <= re_threshold) / re.shape[0]
        for te_threshold in [0.05, 0.1, 0.25, 0.5, 0.75]:
            metrics["TE_{}".format(te_threshold)] = np.sum(te <= te_threshold) / te.shape[0]
        
    gtlo_file = os.path.join(info["gt_file_path"][0], "gtLo.log")
    infolo_file = os.path.join(info["gt_file_path"][0], "gtLo.info")

    if os.path.exists(gtlo_file) and os.path.exists(infolo_file):
        gtlo_pairs, gtlo = read_trajectory(gtlo_file)
        camera_num, gtlo_info = read_trajectory_info(infolo_file)

        transform_pred = []
        for pair in gtlo_pairs:
            id0, id1 = pair[0], pair[1]
            trans = abs_pose_pred[id0] @ inv_transform(abs_pose_pred[id1])
            transform_pred.append(trans)
        transform_pred = np.stack(transform_pred, axis=0)

        precision_lo, recall_lo, flags_lo, errors_lo, re_lo, te_lo, rre_lo, rte_lo = evaluate_registration_RMSE(camera_num, transform_pred, gtlo_pairs, gtlo_pairs, gtlo, gtlo_info, err2)

        metrics["RR_lo"] = recall_lo
        metrics["RE_mean_lo"] = re_lo
        metrics["TE_mean_lo"] = te_lo
        metrics["RRE_lo"] = rre_lo
        metrics["RTE_lo"] = rte_lo

    return metrics