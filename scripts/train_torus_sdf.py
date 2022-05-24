import sys
import os

from options import create_parser, add_training_options, add_model_options

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.meshing import marching_cubes
from ensdf.utils import get_cuda_if_available


def main():
    arg_parser = create_parser()
    training_group = add_training_options(arg_parser)
    model_group = add_model_options(arg_parser)

    # Dataset options
    dataset_group = arg_parser.add_argument_group('Dataset options')
    dataset_group.add_argument('--surface_samples', type=int, default=120_000,
                               help='Number of on surface samples per training iteration.')
    dataset_group.add_argument('--major_radius', type=float, default=0.5,
                               help='The major radius of the torus')
    dataset_group.add_argument('--minor_radius', type=float, default=0.15,
                               help='The minor radius of the torus')

    options = arg_parser.parse_args()

    device = get_cuda_if_available()
    dataset = datasets.TorusDataset(
        major_radius=options.major_radius,
        minor_radius=options.minor_radius,
        num_samples=options.surface_samples,
        device=device
    )
    model = modules.Siren(
        in_features=3, hidden_features=options.hidden_features,
        hidden_layers=options.hidden_layers, out_features=1,
        weight_norm=options.weight_norm,
        first_omega_0=30, outermost_linear=True
    )

    training.train_sdf(
        model=model, surface_dataset=dataset, epochs=options.num_epochs, lr=options.lr,
        epochs_til_checkpoint=options.epochs_til_ckpt, pretrain_epochs=options.pretrain_epochs,
        regularization_samples=options.regularization_samples,
        model_dir=options.model_dir, device=device
    )

    mesh = marching_cubes(model)
    mesh_dir = os.path.join(options.model_dir, 'mesh')
    os.mkdir(mesh_dir)
    mesh.export(os.path.join(mesh_dir, 'mesh.ply'))


if __name__ == '__main__':
    main()
