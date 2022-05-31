import torch
import torch.nn.functional as F
import numpy as np

from .diff_operators import gradient, laplace
from .sampling import sample_uniform_aabb


def triangle_area(triag_verts):
    """
    Computes the area of triangles given by their vertices

    Args:
        triag_verts (torch.Tensor): tensor with shape [*, 3, 3] with the vertices of each triangle
    
    Returns:
        torch.Tensor: tensor with shape [triag_verts.shape[0]] with the area of each triangle
    """
    edge1 = triag_verts[:, 0] - triag_verts[:, 1]
    edge2 = triag_verts[:, 1] - triag_verts[:, 2]

    return torch.norm(torch.cross(edge1, edge2), dim=1) / 2


def normalize_point_cloud(pc, border=0):
    """
    Normalizes a point cloud to lie inside [-(1 - border), 1 - border]^N by centering its bounding box
    to the origin and scaling uniformly using its largest axis as the inverse of the scale

    Args:
        pc (torch.Tensor): The point cloud to normalize with shape [N, N], 
                           where M the number of pointe and  N the number of dimensions
        border (optional float): This should be in [0, 1). Default value is 0.
    """
    pc -= pc.mean(axis=0, keepdims=True)
    max_extent = pc.max() - pc.min()
    pc *= (1 - border) * 2 / max_extent


def normalize_open3d_geometry(geometry, border=0):
    """
    Normalizes the open3d geometry to lie inside [-(1 - border), 1 - border]^3 by centering its bounding box
    to (0,0,0) and scaling uniformly using its largest axis as the inverse of the scale

    Args:
        geometry (open3d.geometry): The geometry to normalize
        border (optional float): This should be in [0, 1). Default value is 0.
    """
    aabb = geometry.get_axis_aligned_bounding_box()
    geometry.translate(-aabb.get_center())
    geometry.scale((1 - border) * 2 / aabb.get_max_extent(), np.zeros((3,1)))


def normalize_trimesh(mesh, border=0):
    """
    Normalizes a trimesh to lie inside [-(1 - border), 1 - border]^3 by centering its bounding box
    to (0,0,0) and scaling uniformly using its largest axis as the inverse of the scale

    Args:
        mesh (trimesh.Trimesh): The geometry to normalize
        border (optional float): This should be in [0, 1). Default value is 0.
    """
    bounds = mesh.bounds
    bounds_center = bounds.mean(0)
    max_extent = mesh.extents.max()
    mesh.apply_translation(-bounds_center)
    mesh.apply_scale((1 - border) * 2 / max_extent)    


def project_on_surface(model, samples, num_steps):
    samples = samples.detach()
    
    for i in range(num_steps):
        samples.requires_grad = True
        sdf_pred = model(samples)
        
        grad = gradient(sdf_pred, samples)
        samples  = samples.detach()
        sdf_pred = sdf_pred.detach()
        grad     = grad.detach()
        
        samples.addcmul_(F.normalize(grad, dim=1), sdf_pred, value=-1)
        
    return samples, sdf_pred, grad


def tangent_grad(grad, normal):
    n_comp = (grad * normal).sum(-1, keepdim=True)
    res = torch.addcmul(grad, n_comp, normal, value=-1)
    res.div_(-n_comp)
    return res


def intersect_plane(plane_normal, plane_center, points, directions):
    denom = (plane_normal * directions).sum(-1, keepdim=True)
    t = ((plane_center - points) * plane_normal).sum(-1, keepdim=True) / denom
    return points + t * directions


def lerp(x0, x1, t):
    y = torch.addcmul(x0, t, x0, value=-1)
    y.addcmul_(t, x1)
    return y


def slerp(x0, x1, t):
    # TODO: fix this, it gives nan
    dot = (x0 * x1).sum(-1)
    theta = torch.acos(dot, out=dot)
    sin_theta = torch.sin(theta)
    m0 = torch.sin((1-t) * theta) / sin_theta
    res = m0.unsqueeze(-1) * x0
    m1 = torch.sin(t * theta) / sin_theta
    res.addcmul_(m1.unsqueeze(-1), x1)
    return res


def linear_fall(x):
    y = 1 - abs(x)
    y = F.relu(y)
    return y


def cubic_fall(x):
    x = linear_fall(x)
    x_sq = x**2
    y = 3 * x_sq - 2 * x_sq * x
    return y


def quintic_fall(x):
    x = linear_fall(x)
    y = x**3 * (x * (x * 6.0 - 15.0) + 10.0)
    return y

def exp_fall(x):
    t = 1 - x**2
    t = F.relu(t)
    y = torch.exp(1 - 1 / t)
    return y


def euler_to_matrix(yaw, pitch, roll):
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)

    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)

    c_roll = np.cos(roll)
    s_roll = np.sin(roll)    

    yaw_mat = np.array(
        [
            [c_yaw , 0., s_yaw],
            [0.    , 1., 0.   ],
            [-s_yaw, 0., c_yaw]
        ]
    )
    pitch_mat = np.array(
        [
            [1., 0.     , 0.      ],
            [0., c_pitch, -s_pitch],
            [0., s_pitch, c_pitch ]
        ]
    )
    roll_mat = np.array(
        [
            [c_roll, -s_roll, 0.],
            [s_roll, c_roll , 0.],
            [0.    , 0.     , 1.]
        ]
    )

    return yaw_mat @ pitch_mat @ roll_mat


def spherical_to_cartesian(phi, theta, radius):
    # y is up
    y = radius * np.cos(theta)
    x = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi) * np.sin(theta)

    return x, y, z


def sdf_volume(model, aabb, num_samples):
    samples = sample_uniform_aabb(aabb, num_samples)
    
    sdf = model(samples)
    inside_surface = sdf <= 0
    
    volume = torch.count_nonzero(inside_surface) / num_samples * aabb.volume()
    
    return volume


def sdf_area(model, aabb, num_samples):
    samples = sample_uniform_aabb(aabb, num_samples)
    samples.requires_grad = True
    
    sdf = model(samples)
    inside_surface = sdf <= 0
    
    volume = torch.count_nonzero(inside_surface) / num_samples * aabb.volume()
    
    # Divergence of gradient
    divergence = laplace(sdf, samples)
    divergence = divergence[inside_surface].detach()

    area = divergence.mean() * volume

    return area
