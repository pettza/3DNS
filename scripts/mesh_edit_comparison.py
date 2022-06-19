import sys
import os
from pyrsistent import b

import torch
import trimesh

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


def main():
    arg_parser = create_edit_parser()

    # Mesh options
    mesh_group = arg_parser.add_argument_group('Mesh options')
    mesh_group.add_argument('--mesh_path', type=str,
                            help='Path to the mesh that the network was trained on.')
    
    options = arg_parser.parse_args()

    device = get_cuda_if_available()
    model = modules.Siren.load(options.model_path)
    model.to(device)
    model_size = model.get_num_bytes()

    if options.mesh_path:
        original_mesh = trimesh.load_mesh(options.mesh_path)
        normalize_trimesh(original_mesh, border=0.15)
    else:
        original_mesh = marching_cubes(model)

    original_num_faces = original_mesh.faces.shape[0]
    original_size = size_of_trimesh(original_mesh)
    simple_num_faces, simple_mesh, simple_size = simplify_trimesh(original_mesh, model_size)

    print(f'Size of model: {model_size}')
    print(f'Number of faces and size of original mesh: {original_num_faces} {original_size}')
    print(f'Number of faces and size of simple mesh: {simple_num_faces} {simple_size}')

    aabb = AABB([0., 0., 0.], [1., 1., 1.], device=device)

    origin    = torch.tensor([[options.ox, options.oy, options.oz]], device=device)
    direction = torch.tensor([[options.dx, options.dy, options.dz]], device=device)

    inter_point, inter_normal, inter_sdf, ray_hit = raymarch_single_ray(model, aabb, origin, direction)
    
    if not ray_hit:
        print("The specified ray doesn't intersect the surface")
        exit()

    brush = SimpleBrush(
        brush_type=options.brush_type,
        radius=options.brush_radius,
        intensity=options.brush_intensity
    )
    brush.set_interaction(inter_point, inter_normal, inter_sdf)

    # Edit model
    dataset = datasets.SDFEditingDataset(
        model, device, brush,
        num_interaction_samples=options.num_interaction_samples,
        num_model_samples=options.num_model_samples
    )

    training.train_sdf(
        model=model, surface_dataset=dataset, epochs=options.num_epochs, lr=options.lr,
        epochs_til_checkpoint=options.num_epochs, pretrain_epochs=0,
        regularization_samples=options.regularization_samples,
        include_empty_space_loss=not options.no_empty_space,
        ewc=options.ewc, model_dir=options.model_dir, device=device
    )

    # Edit meshes
    original_mesh_edited = brush.deform_mesh(original_mesh)
    simple_mesh_edited = brush.deform_mesh(simple_mesh)
    # original_mesh_edited = original_mesh
    # simple_mesh_edited = simple_mesh

    # Save meshes
    oritinal_mesh_edtied_path = os.path.join(options.model_dir, 'original_mesh_edited.obj')
    simple_mesh_edited_path = os.path.join(options.model_dir, 'simple_mesh_edited.obj')
    
    original_mesh_edited.export(oritinal_mesh_edtied_path)
    simple_mesh_edited.export(simple_mesh_edited_path)

    chamfer_samples = 100_000
    original_mesh_edited_dataset = datasets.MeshDataset(oritinal_mesh_edtied_path, chamfer_samples, device)
    simple_mesh_edited_dataset = datasets.MeshDataset(simple_mesh_edited_path, chamfer_samples, device)
    model_sampler = SDFSampler(model, device, chamfer_samples, burnout_iters=100)
    
    original_mesh_editied_pc = original_mesh_edited_dataset.sample()['points'].cpu().numpy()
    simple_mesh_edited_pc = simple_mesh_edited_dataset.sample()['points'].cpu().numpy()
    model_pc = next(model_sampler)['points'].cpu().numpy()

    model_dist = chamfer(original_mesh_editied_pc, model_pc)
    simple_mesh_dist = chamfer(original_mesh_editied_pc, simple_mesh_edited_pc)

    with open(os.path.join(options.model_dir, 'chamfer_distances'), 'w') as f:
        f.write('Distance between edited model and edited original mesh:\n')
        f.write(f'\t{model_dist}\n')
        f.write('Distance between edited simple mesh and edited original mesh:\n')
        f.write(f'\t{simple_mesh_dist}\n')

    print('Distance between edited model and edited original mesh:')
    print(f'\t{model_dist}')
    print('Distance between edited simple mesh and edited original mesh:')
    print(f'\t{simple_mesh_dist}')


if __name__ == '__main__':
    main()
