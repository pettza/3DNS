import sys
import os

import torch
import numpy as np
import trimesh
from trimesh.smoothing import filter_humphrey
import matplotlib.pyplot as plt

from options import create_parser

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import modules
from ensdf.sampling.sdf import SDFSampler
from ensdf.utils import get_cuda_if_available
from ensdf.meshing import marching_cubes, ball_pivoting
from ensdf.geoutils import triangle_area


def get_mesh(options, sampler):
    if options.mesh_path:
        mesh = trimesh.load_mesh(options.mesh_path)
        return mesh
    
    if options.ball_pivoting:
        # Samples to construct reference mesh
        samples = next(sampler)
        points = samples['points'].detach().cpu().numpy()
        normals = torch.nn.functional.normalize(
            samples['normals'].detach()
        ).cpu().numpy()
        mesh = ball_pivoting(points, normals)
        return mesh

    # If neither then marching cubes was selected
    mesh = marching_cubes(sampler.model, options.marching_cubes)
    filter_humphrey(mesh)
    return mesh


def main():
    arg_parser = create_parser()

    # Model options
    arg_parser.add_argument('--checkpoint_path', type=str, required=True,
                            help='Path to model checkpoint')

    # Method options
    arg_parser.add_argument('--num_iterations', type=int, default=100,
                            help='Number of iterations for estimation')

    # Sampler options
    sampler_group = arg_parser.add_argument_group('Samples options')
    sampler_group.add_argument('--num_samples', type=int, default=10000,
                            help='Number of samples per iteration for SDF sampler')
    sampler_group.add_argument('--burnout_iterations', type=int, default=10,
                            help='Number of iterations before starting using them for estimation')

    # Mesh options
    mesh_arg_group = arg_parser.add_argument_group('Mesh options')
    mesh_mutex_group = mesh_arg_group.add_mutually_exclusive_group(required=True)
    mesh_mutex_group.add_argument('--mesh_path', type=str,
                                help='Path to the mesh file.')
    mesh_mutex_group.add_argument('--ball_pivoting', action='store_true',
                                help='Uses ball pivoting')
    mesh_mutex_group.add_argument('--marching_cubes', type=int,
                                help='Uses marching cubes with this grid resolution')
    
    options = arg_parser.parse_args()

    device = get_cuda_if_available()

    model = modules.Siren.load(options.checkpoint_path)
    model.to(device)

    sampler = SDFSampler(
        model, device,
        num_samples=options.num_samples,
        burnout_iters=options.burnout_iterations
    )

    mesh = get_mesh(options, sampler)

    # Compute triangle areas
    vertices = mesh.vertices
    triangles = mesh.faces
    triagle_verts = vertices[triangles]
    areas = triangle_area(torch.from_numpy(triagle_verts)).numpy()

    # Class for finding closest triangles
    prox = trimesh.proximity.ProximityQuery(mesh)
    
    # Compute histogram on triangles
    hist = np.zeros((triangles.shape[0]))
    for i in range(options.num_iterations):
        points = next(sampler)['points'].detach().cpu().numpy()

        closest, dist, triangle_ids = prox.on_surface(points)
        
        unique_ind, counts = np.unique(triangle_ids, return_counts=True)
        hist[unique_ind] += counts

    # Normalize densities with total number of samples and triagle area
    densities = hist / (options.num_samples * options.num_iterations * areas)

    # Compute colors based on densities
    triangle_colors = plt.get_cmap('plasma')(densities)
    triangle_colors = triangle_colors[:, :3] # Remove alpha channel

    mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=triangle_colors)
    mesh.show()

    inv_area = 1. / mesh.area
    print(f'Inverse mesh area: {inv_area}')

    print(f'Number of triangles: {hist.shape[0]}')
    fig, ax = plt.subplots()
    ax.hist(densities, bins='auto')
    ax.axvline(x=inv_area, color='red', linestyle='--')
    ax.set_xticks(ax.get_xticks() + [inv_area])
    plt.show()


if __name__ == '__main__':
    main()
