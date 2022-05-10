from matplotlib.pyplot import get
import torch
import numpy as np
import skimage.measure
import trimesh

from ..utils import get_cuda_if_available


def marching_cubes(model, N=256, max_batch=64**3):
    device = get_cuda_if_available()
    model.to(device)
    model.eval()

    num_samples = N ** 3

    s = torch.linspace(-1, 1, N)
    samples = torch.stack(torch.meshgrid(s, s, s, indexing='ij'), dim=-1).reshape(-1, 3)
    sdf_values = torch.empty(num_samples)

    with torch.no_grad():
        for head in range(0, num_samples, max_batch):
            sample_subset = samples[head : min(head + max_batch, num_samples)].to(device)

            sdf_values[head : min(head + max_batch, num_samples)] = (
                model(sample_subset).squeeze().cpu()
            )
            head += max_batch
    
    sdf_values = sdf_values.reshape(N, N, N).numpy()

    voxel_size = 2. / (N - 1)
    vertices, faces, normals, values = skimage.measure.marching_cubes(
        sdf_values, level=0.0, spacing=[voxel_size] * 3
    )
    # Translate vertices
    vertices -= [1., 1., 1.]

    return trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
