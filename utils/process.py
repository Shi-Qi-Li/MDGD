import numpy.typing as npt
import numpy as np
import torch


def square_distance(pc_1, pc_2):
    assert pc_1.shape[-1] == pc_2.shape[-1]
    
    if isinstance(pc_1, np.ndarray):
        pc_1 = torch.from_numpy(pc_1)
    if isinstance(pc_2, np.ndarray):
        pc_2 = torch.from_numpy(pc_2)

    if torch.cuda.is_available():
        pc_1 = pc_1.cuda()
        pc_2 = pc_2.cuda()

    dist = -2 * torch.matmul(pc_1, pc_2.transpose(-1, -2))
    dist += torch.sum(pc_1 ** 2, dim=-1, keepdim=True)
    dist += torch.sum(pc_2 ** 2, dim=-1, keepdim=True).transpose(-1, -2)

    return dist.cpu()

def random_rotation_matrix(rotrange = 180):
    """
    Generates a random 3D rotation matrix from axis and angle.

    Args:
        numpy_random_state: numpy random state object

    Returns:
        Random rotation matrix.
    """
    rng = np.random.RandomState()
    axis = rng.rand(3) - 0.5
    axis /= np.linalg.norm(axis) + 1E-8
    theta = rotrange / 180 * np.pi * rng.uniform(0.0, 1.0)
    thetas = axis * theta
    alpha = thetas[0]
    beta = thetas[1]
    gama = thetas[2]
    Rzalpha = np.array([[np.cos(alpha),np.sin(alpha),0],
                      [-np.sin(alpha),np.cos(alpha),0],
                      [0,0,1]])

    Rybeta = np.array([[np.cos(beta),0,-np.sin(beta)],
                     [0,1,0],
                     [np.sin(beta),0,np.cos(beta)]])

    Rzgama = np.array([[np.cos(gama),np.sin(gama),0],
                      [-np.sin(gama),np.cos(gama),0],
                      [0,0,1]])
    
    R = np.matmul(Rzgama, np.matmul(Rybeta, Rzalpha))
    return R

def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T
    
def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4, dtype=torch.float32)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.repeat(np.eye(4, dtype=np.float32)[None], R.shape[0], axis=0)
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.reshape(-1, 3, 1)
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4, dtype=torch.float32).to(R.device)
        else:
            trans = np.eye(4, dtype=np.float32)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans

def inv_transform(transform):
    if len(transform.shape) == 3:
        if isinstance(transform, torch.Tensor):
            inv_transform = torch.eye(4, dtype=torch.float32)[None].repeat(transform.shape[0], 1, 1).to(transform.device)
            inv_transform[:, :3, :3] = transform[:, :3, :3].permute(0, 2, 1)
            inv_transform[:, :3, 3:] = torch.bmm(-transform[:, :3, :3].permute(0, 2, 1), transform[:, :3, 3:])
        else:
            inv_transform = np.repeat(np.eye(4, dtype=np.float32)[None], transform.shape[0], axis=0)
            inv_transform[:, :3, :3] = transform[:, :3, :3].swapaxes(-1, -2)
            inv_transform[:, :3, 3:] = -transform[:, :3, :3].swapaxes(-1, -2) @ transform[:, :3, 3:]
    else:
        if isinstance(transform, torch.Tensor):
            inv_transform = torch.eye(4, dtype=torch.float32).to(transform.device)
        else:
            inv_transform = np.eye(4, dtype=np.float32)
        inv_transform[:3, :3] = transform[:3, :3].T
        inv_transform[:3, 3:] = -transform[:3, :3].T @ transform[:3, 3:]
    return inv_transform