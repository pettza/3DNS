import sys
import os

import torch

from options import create_parser

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from ensdf import modules
from ensdf.meshing import marching_cubes, ball_pivoting
from ensdf.sampling.sdf import SDFSampler
from ensdf.utils import get_cuda_if_available


def main():
    arg_parser = create_parser()
    arg_parser.add_argument('--mesh_path', type=str,
                            help='The name of the output file')
    arg_parser.add_argument('--show', action='store_true',
                            help='Show the mesh before exitng')

    # Model options
    model_group = arg_parser.add_argument_group('Model options')
    model_group.add_argument('--checkpoint_path', type=str, required=True,
                             help='Path to model checkpoint')

    # Method options
    method_arg_group = arg_parser.add_argument_group('Meshing method options')
    method_mutex_group = method_arg_group.add_mutually_exclusive_group(required=True)
    method_mutex_group.add_argument(
        '--ball_pivoting', type=int,
        help='Uses ball pivoting wiht this number of samples')
    method_mutex_group.add_argument(
        '--marching_cubes', type=int,
        help='Uses marching cubes with this grid resolution')
    
    options = arg_parser.parse_args()
    
    device = get_cuda_if_available()
    model = modules.Siren.load(options.checkpoint_path)
    model.to(device)

    if options.ball_pivoting:
        sampler = SDFSampler(model, device, num_samples=options.ball_pivoting)
        
        # Burnout
        samples = next(sampler)
        for i in range(10):
            samples = next(sampler)

        points = samples['points'].detach().cpu().numpy()
        normals = torch.nn.functional.normalize(samples['normals'].detach()).cpu().numpy()

        mesh = ball_pivoting(points, normals)
    else: # Marching cubes
        mesh = marching_cubes(model, N=options.marching_cubes, max_batch=128**3)

    mesh.export(options.mesh_path)

    if options.show:
        mesh.show()


if __name__ == '__main__':
    main()
