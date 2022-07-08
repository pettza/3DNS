import sys
import os
import copy

import torch
import numpy as np
import trimesh
import shutil

from options import create_edit_parser

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.aabb import AABB
from ensdf.raymarching import raymarch_single_ray
from ensdf.brushes import SimpleBrush
from ensdf.geoutils import normalize_trimesh
from ensdf.meshing import marching_cubes
from ensdf.utils import get_cuda_if_available, simplify_trimesh, size_of_trimesh
from ensdf.metrics import chamfer
from ensdf.sampling.sdf import SDFSampler


def gather_samples_in_interaction(brush, sample_fn, n_samples):
    gathered_samples = 0
    samples = []
    while gathered_samples < n_samples:
        iter_samples = sample_fn()
        cond = brush.inside_interaction(iter_samples)
        samples.append(iter_samples[cond])
        gathered_samples += samples[-1].shape[0]

    samples = torch.concat(samples)[:n_samples]
    return samples


def main():
    arg_parser, arg_groups = create_edit_parser()

    # Mesh options
    mesh_group = arg_parser.add_argument_group('Mesh options')
    mesh_group.add_argument('--mesh_path', type=str,
                            help='Path to the mesh that the network was trained on.')
    
    # Comparison options
    comparison_group = arg_parser.add_argument_group('Mesh options')
    comparison_group.add_argument('--chamfer_samples', type=float, default=100_000,
                                  help='The number of points to be used for chamfer distance')
    comparison_group.add_argument('--average', type=int, default=None,
                                  help='If specified, the average of multiple random edits \
                                        will be computed')
    comparison_group.add_argument('--num_edits', type=int, default=None,
                                  help='If specified, the edit from interaction options \
                                        will be ignored and instead the given number of points \
                                        will be randomly chosen on the surface. This will happen \
                                        per iteration if --average is also specified.')

    options = arg_parser.parse_args()

    if os.path.exists(options.model_dir):
        prompt = f'The directory {options.model_dir} exists. Overwrite? (y/n): '
        
        val = input(prompt).lower()
        while val not in {'y', 'n'}:
            val = input(prompt).lower()
        
        if val == 'y':
            shutil.rmtree(options.model_dir)
        else:
            print('Cannot proceed without valid directory')
            exit()

    os.makedirs(options.model_dir, exist_ok=True)

    device = get_cuda_if_available()
    model = modules.Siren.load(options.model_path)
    model.to(device)
    model_size = model.get_num_bytes()

    if options.mesh_path:
        print(f'Reading mesh from {options.mesh_path}...')
        original_mesh = trimesh.load_mesh(options.mesh_path)
        normalize_trimesh(original_mesh, border=0.15)
        print(f'Done')
    else:
        print('Creating mesh...')
        original_mesh = marching_cubes(model)
        print('Done')

    print('Simplifying mesh...')
    original_num_faces = original_mesh.faces.shape[0]
    original_size = size_of_trimesh(original_mesh)
    simple_num_faces, simple_mesh, simple_size = simplify_trimesh(original_mesh, model_size)
    print('Done')

    print(f'Size of model: {model_size}')
    print(f'Number of faces and size of original mesh: {original_num_faces} {original_size}')
    print(f'Number of faces and size of simple mesh: {simple_num_faces} {simple_size}')

    brush = SimpleBrush(
        brush_type=options.brush_type,
        radius=options.brush_radius,
        intensity=options.brush_intensity
    )

    random_edits = options.average or options.num_edits
    num_edits = options.num_edits or 1

    edit_iterations = options.average or 1
    model_total_dists = np.empty(edit_iterations)
    model_inter_dists = np.empty((edit_iterations, num_edits))
    simple_mesh_total_dists = np.empty(edit_iterations)
    simple_mesh_inter_dists = np.empty((edit_iterations, num_edits))

    for i in range(edit_iterations):
        model_copy = copy.deepcopy(model)
        original_mesh_edited = copy.deepcopy(original_mesh)
        simple_mesh_edited = copy.deepcopy(simple_mesh)

        model_sampler = SDFSampler(model_copy, device, options.chamfer_samples, burnout_iters=100)

        # Edit model
        dataset = datasets.SDFEditingDataset(
            model_copy, device, brush,
            num_model_samples=options.num_model_samples,
            interaction_samples_factor=options.interaction_samples_factor
        )

        if random_edits:
            samples = next(model_sampler)
            rnd_ind = torch.randint(0, model_sampler.num_samples, (num_edits,))
            inter_points = samples['points'][rnd_ind]
            inter_normals = samples['normals'][rnd_ind]
            inter_sdfs = samples['sdf'][rnd_ind]
        else:
            aabb = AABB([0., 0., 0.], [1., 1., 1.], device=device)

            origin    = torch.tensor([[options.ox, options.oy, options.oz]], device=device)
            direction = torch.tensor([[options.dx, options.dy, options.dz]], device=device)

            inter_points, inter_normals, inter_sdfs, ray_hit = raymarch_single_ray(model, aabb, origin, direction)
        
            if not ray_hit:
                print("The specified ray doesn't intersect the surface")
                exit()
            
        for j, (inter_point, inter_normal, inter_sdf) in enumerate(zip(inter_points, inter_normals, inter_sdfs)):
            brush.set_interaction(
                inter_point.unsqueeze(0),
                inter_normal.unsqueeze(0),
                inter_sdf.unsqueeze(0)
            )

            training.train_sdf(
                model=model_copy, surface_dataset=dataset, epochs=options.num_epochs, lr=options.lr,
                epochs_til_checkpoint=options.num_epochs, pretrain_epochs=0,
                regularization_samples=options.regularization_samples,
                include_empty_space_loss=not options.no_empty_space,
                ewc=options.ewc, device=device
            )
            dataset.update_model(model_copy, sampler_iters=20)
            model_sampler.burnout(20)

            # Edit meshes
            original_mesh_edited = brush.deform_mesh(original_mesh_edited)
            simple_mesh_edited = brush.deform_mesh(simple_mesh_edited)

            if not options.average:
                # Save meshes and model
                original_mesh_edtied_path = os.path.join(options.model_dir, 'original_mesh_edited.obj')
                simple_mesh_edited_path = os.path.join(options.model_dir, 'simple_mesh_edited.obj')
                model_path = os.path.join(options.model_dir, 'model.pth')

                original_mesh_edited.export(original_mesh_edtied_path)
                simple_mesh_edited.export(simple_mesh_edited_path)
                model_copy.save(model_path)

            original_mesh_edited_dataset = datasets.MeshDataset(
                original_mesh_edited, options.chamfer_samples, device, normalize=False
            )
            simple_mesh_edited_dataset = datasets.MeshDataset(
                simple_mesh_edited, options.chamfer_samples, device, normalize=False
            )
    
            # Chamfer distance over interaction
            brush.radius = max(brush.radius, brush.intensity) + 0.02 # Increase radius a bit
            original_mesh_editied_inter_pc = gather_samples_in_interaction(
                brush, lambda: original_mesh_edited_dataset.sample()['points'], options.chamfer_samples
            ).cpu().numpy()
            simple_mesh_edited_inter_pc = gather_samples_in_interaction(
                brush, lambda: simple_mesh_edited_dataset.sample()['points'], options.chamfer_samples
            ).cpu().numpy()
            model_inter_pc = gather_samples_in_interaction(
                brush, lambda: next(model_sampler)['points'], options.chamfer_samples
            ).cpu().numpy()
            brush.radius = options.brush_radius # Revert to specified radius

            model_inter_dist = chamfer(original_mesh_editied_inter_pc, model_inter_pc)
            simple_mesh_inter_dist = chamfer(original_mesh_editied_inter_pc, simple_mesh_edited_inter_pc)
            model_inter_dists[i][j] = model_inter_dist
            simple_mesh_inter_dists[i][j] = simple_mesh_inter_dist

        original_mesh_edited_dataset = datasets.MeshDataset(
            original_mesh_edited, options.chamfer_samples, device, normalize=False
        )
        simple_mesh_edited_dataset = datasets.MeshDataset(
            simple_mesh_edited, options.chamfer_samples, device, normalize=False
        )
    
        # Chamfer distance over entire surface
        original_mesh_editied_total_pc = original_mesh_edited_dataset.sample()['points'].cpu().numpy()
        simple_mesh_edited_total_pc = simple_mesh_edited_dataset.sample()['points'].cpu().numpy()
        model_total_pc = next(model_sampler)['points'].cpu().numpy()

        model_total_dist = chamfer(original_mesh_editied_total_pc, model_total_pc)
        simple_mesh_total_dist = chamfer(original_mesh_editied_total_pc, simple_mesh_edited_total_pc)
        model_total_dists[i] = model_total_dist
        simple_mesh_total_dists[i] = simple_mesh_total_dist

    def write_distances(f, dists):
        if dists.size > 1:
            f.write(f'\tMean: {dists.mean()}\n')
            f.write(f'\tStd: {dists.std()}\n')
        else:
            f.write(f'\t{dists.item(0)}\n')
    
    results_filename = os.path.join(options.model_dir, 'chamfer_distances.txt')
    with open(results_filename, 'w') as f:
        f.write('- Chamfer distance over entire surface\n')
        f.write(f'Number of samples: {model_total_dists.size}\n')
        f.write('Model - Original Mesh:\n')
        write_distances(f, model_total_dists)
        f.write('Simple Mesh - Original Mesh:\n')
        write_distances(f, simple_mesh_total_dists)
        f.write('Relative distances:\n')
        relative_total_dists = (simple_mesh_total_dists - model_total_dists) / simple_mesh_total_dists
        write_distances(f, relative_total_dists)

        f.write('\n')

        f.write('- Chamfer distance over interaction\n')
        f.write('Model - Original Mesh:\n')
        f.write(f'Number of samples: {model_inter_dists.size}\n')
        write_distances(f, model_inter_dists)
        f.write('Simple Mesh - Original Mesh:\n')
        write_distances(f, simple_mesh_inter_dists)
        f.write('Relative distances:\n')
        relative_inter_dists = (simple_mesh_inter_dists - model_inter_dists) / simple_mesh_inter_dists
        write_distances(f, relative_inter_dists)

    with open(results_filename) as f:
        print(f.read())


if __name__ == '__main__':
    main()
