import sys
import os

import torch

from options import create_parser, add_training_options, add_model_options
from ensdf.utils import get_cuda_if_available

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.utils import get_cuda_if_available


def main():
    arg_parser = create_parser()
    training_group = add_training_options(arg_parser)
    model_group = add_model_options(arg_parser)

    # Dataset options
    dataset_group = arg_parser.add_argument_group('Dataset options')
    dataset_group.add_argument('--surface_samples', type=int, default=120_000,
                               help='Number of on surface samples per training iteration.')
    dataset_group.add_argument('--radius', type=float, default=0.8,
                               help='The radius of the sphere. Should be < 1.')

    options = arg_parser.parse_args()

    device = get_cuda_if_available()
    dataset = datasets.SphereDataset(
        radius=options.radius,
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


if __name__ == '__main__':
    main()
