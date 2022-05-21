import sys
import os

import torch

from options import create_parser, add_edit_training_options

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.aabb import AABB
from ensdf.raymarching import raymarch_single_ray
from ensdf.utils import get_cuda_if_available
from ensdf.brushes import SimpleBrush


def main():
    arg_parser = create_parser()
    add_edit_training_options(arg_parser)


    # Model options
    model_group = arg_parser.add_argument_group('Model options')
    model_group.add_argument('--model_path', type=str, required=True,
                             help='Path to pretrained model.')


    # Dataset options
    dataset_group = arg_parser.add_argument_group('Dataset options')
    dataset_group.add_argument('--num_interaction_samples', type=int, default=5000,
                               help='Number of samples for interaction.')
    dataset_group.add_argument('--num_model_samples', type=int, default=120000,
                               help='Number of samples from pretrained model.')


    # Interaction options
    interaction_group = arg_parser.add_argument_group('Interaction options')
    interaction_group.add_argument('--ox', type=float, default=0.,
                                   help='X component of origin of interaction ray')
    interaction_group.add_argument('--oy', type=float, default=0.,
                                   help='Y component of origin of interaction ray')
    interaction_group.add_argument('--oz', type=float, default=0.9,
                                   help='Z component of origin of interaction ray')

    interaction_group.add_argument('--dx', type=float, default=0.,
                                   help='X component of direction of interaction ray')
    interaction_group.add_argument('--dy', type=float, default=0.,
                                   help='Y component of direction of interaction ray')
    interaction_group.add_argument('--dz', type=float, default=-1.,
                                   help='Z component of direction of interaction ray')

    interaction_group.add_argument('--brush_radius', type=float, default=0.08,
                                    help='The radius of the brush')
    interaction_group.add_argument('--brush_intensity', type=float, default=0.03,
                                    help='The intensity of the brush')
    interaction_group.add_argument('--brush_type', choices=['linear', 'cubic', 'quintic'], default='quintic',
                                    help='The type of the brush')
    

    options = arg_parser.parse_args()

    device = get_cuda_if_available()
    model = modules.Siren.load(options.model_path)
    model.to(device)

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


if __name__ == '__main__':
    main()
