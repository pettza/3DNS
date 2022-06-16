import sys
import os

import torch

from options import create_edit_parser

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.aabb import AABB
from ensdf.raymarching import raymarch_single_ray
from ensdf.utils import get_cuda_if_available
from ensdf.brushes import SimpleBrush


def main():
    arg_parser = create_edit_parser()

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
