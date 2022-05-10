import torch
import torch.nn.functional as F


def sample_uniform_sphere(num_samples, device):
    '''
    Uniformly samples unit sphere using Muller technique (normalizing 3d normal distributed samples)

    Args:
        num_samples (int): the number of samples to return
        device (torch.device): the device of the returned tensor

    Returns:
        torch.Tensor: The sampled points with shape [num_samples, 3]
    '''
    x = torch.randn(num_samples, 3, device=device)
    F.normalize(x, dim=1, out=x)
    return x


def sample_uniform_ball(num_samples, device):
    '''
    Uniformly samples unit ball using muller technique

    Args:
        num_samples (int): the number of samples to return
        device (torch.device): the device of the returned tensor

    Returns:
        torch.Tensor: The sampled points with shape [num_samples, 3]
    '''
    x = sample_uniform_sphere(num_samples, device)
    r = torch.pow(torch.rand(num_samples, device=device), 1/3)
    x.mul_(r)
    return x


def sample_uniform_disk(normals, samples_per_normal):
    '''
    Uniformly samples unit discs oriented according to the provided normals

    Args:
        normals (torch.Tensor): tensor with shape [N, 3] with the normals to the disks
        samples_per_normal (int): the number of samples per normal
    
    Returns:
        torch.Tensor: The sampled points with shape [N, samples_per_normal, 3]
    '''
    # Sample 4d sphere and discard 2 dimensions
    # the first one is discarded by omitting the last dimension
    # the second one by removing the component along the normal
    x = torch.randn(normals.shape[0], samples_per_normal, 4, device=normals.device)
    F.normalize(x, dim=2, out=x)
    x = x[..., :3]
    
    normals = normals.view(-1, 1, 3)
    
    n_comp = (x * normals).sum(-1, keepdim=True)
    x.addcmul_(n_comp, normals, value=-1)
    return x


def sample_planar_gaussian(normals, samples_per_normal):
    '''
    Samples normal distribution on planes oriented according to the provided normals

    Args:
        normals (torch.Tensor): tensor with shape [N, 3] with the normals to the disks
        samples_per_normal (int): the number of samples per normal

    Returns:
        torch.Tensor: The sampled points with shape [N, samples_per_normal, 3]
    '''
    # Samples 3d normal and discard component along normal
    x = torch.randn(normals.shape[0], samples_per_normal, 3, device=normals.device)
    
    normals = normals.view(-1, 1, 3)
    
    n_comp = (x * normals).sum(-1).unsqueeze(-1)
    x.addcmul_(n_comp, normals, value=-1)
    return x


def sample_uniform_triangle(triag_verts, normals, samples_per_triag):
    '''
    Uniformly samples triangles given by their vertices
    
    Args:
        triag_verts (torch.Tensor): tensor with shape [N, 3, 3] with the vertices of the triangles
        normals (torch.Tensor): the per vertex normals (same shape as triag_verts)
        samples_per_triag (int): the number of samples per triangle
    
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing:
            - the sampled positions with shape [N, samples_per_triag, 3]
            - the sampled normals with shape [N, samples_per_triag, 3]
            - the barycentric coordinates of the samples with shape [N, samples_per_triag, 2]
    '''
    triag_num = triag_verts.shape[0]

    # Uniformily sample barycentric coordinates
    uv = torch.rand(triag_num, samples_per_triag, 2, device=triag_verts.device)
    uv[..., 0].sqrt_()
    uv[..., 1].mul_(uv[..., 0])
    torch.add(1, uv[..., 0], alpha=-1, out=uv[..., 0])

    # Interpolate vertex positions
    triag_verts = triag_verts.unsqueeze(1)
    samples = uv[..., [0]]*triag_verts[:, :, 0] + \
              uv[..., [1]]*triag_verts[:, :, 1] + \
              (1 - uv.sum(dim=-1, keepdim=True))*triag_verts[:, :, 2]

    # Interpolate normals
    normals = normals.unsqueeze(1)
    normal_samples = uv[..., [0]] * normals[:, :, 0] + \
                        uv[..., [1]] * normals[:, :, 1] + \
                        (1 - uv.sum(dim=-1, keepdim=True)) * normals[:, :, 2]
    return samples, F.normalize(normal_samples, dim=-1), uv


def sample_uniform_mesh(vertices, triangles, normals, area_dist, num_samples):
    '''
    Uniformly samples a triangle mesh

    Args:
        vertices (torch.Tensor): tensor with shape [N, 3] with the vertices of the mesh
        triangles (torch.Tensor): tensor with shape [M, 3] with the vertex indices for each triangle
        normals (otional torch.Tensor): tensor with shape [N, 3] with the per vertex normals
        area_dist (torch.distributions.Categorical): distribution on the triangle
        num_samples (int): the number of samples

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing:
            - the sampled positions with shape [num_samples, 3]
            - the sampled normals with shape [num_samples, 3]
    '''
    triags = area_dist.sample([num_samples])
    face_idx = triangles[triags]
    triag_verts = vertices[face_idx]

    point_samples, normal_samples, uv = sample_uniform_triangle(triag_verts, samples_per_triag=1, normals=normals[face_idx])
    return point_samples.squeeze(1), normal_samples.squeeze(1)
