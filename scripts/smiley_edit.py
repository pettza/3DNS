import sys
import os

import torch
import numpy as np

from options import create_parser

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
from ensdf import datasets, training, modules
from ensdf.aabb import AABB
from ensdf.raymarching import raymarch_single_ray
from ensdf.utils import get_cuda_if_available
from ensdf.brushes import SimpleBrush
from ensdf.geoutils import spherical_to_cartesian


def main():
    arg_parser = create_parser()
    arg_parser.add_argument('--sphere_model_path', type=str, required=True,
                            help='Path to pretrained sphere model.')

    options = arg_parser.parse_args()

    device = get_cuda_if_available()
    model = modules.Siren.load(options.sphere_model_path)
    model.to(device)
    aabb = AABB([0., 0., 0.], [1., 1., 1.], device=device)
    brush = SimpleBrush(
        brush_type='quintic',
        radius=0.08,
        intensity=0.04
    )
    dataset = datasets.SDFEditingDataset(
        model, device, brush,
        num_interaction_samples=5000,
        num_model_samples=120_000
    )
    num_reg_samples = 12000
    lr = 1e-4
    num_epochs = 100

    def trace_from_spherical_set_interaction(phi, theta):
        coords = spherical_to_cartesian(phi, theta, 1)
        origin = torch.tensor([coords], dtype=torch.float32, device=device)
        direction = -origin
        inter_point, inter_normal, inter_sdf, ray_hit = raymarch_single_ray(model, aabb, origin, direction)
        brush.set_interaction(inter_point, inter_normal, inter_sdf)

    def do_interaction(phi, theta):
        trace_from_spherical_set_interaction(phi, theta)
        training.train_sdf(
            model=model, surface_dataset=dataset, epochs=num_epochs, lr=lr,
            regularization_samples=num_reg_samples, device=device
        )
        dataset.update_model(model)

    # Eyes
    phis = [np.deg2rad(25), -np.deg2rad(25)]
    theta = np.deg2rad(60)
    for phi in phis:
        brush.radius = 0.1
        brush.intensity = 0.06
        do_interaction(phi, theta)

    # Nose
    phi = 0
    theta = np.deg2rad(85)
    brush.radius = 0.08
    brush.intensity = 0.05
    do_interaction(phi, theta)

    # Mouth
    mouth_points = 15
    t = np.linspace(-1., 1., mouth_points)
    phis = np.deg2rad(40 * t)
    thetas = np.deg2rad(100 + 20 * np.cos(np.pi/2 * t))
    brush.radius = 0.07
    brush.intensity = 0.04
    for phi, theta in zip(phis, thetas):
         do_interaction(phi, theta)

    model.save('smiley.pth')

if __name__ == '__main__':
    main()
