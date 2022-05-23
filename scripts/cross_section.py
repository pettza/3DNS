import sys
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from options import create_parser

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
import ensdf.modules as modules
from ensdf.rendering.cross_section import cross_section
from ensdf.utils import get_cuda_if_available


def main():
    arg_parser = create_parser()
    arg_parser.add_argument('--model_path', type=str, required=True,
                             help='Path to pretrained model.')

    plane_group = arg_parser.add_argument_group('Plane options')    
    plane_group.add_argument('--nx', type=float, default=0.,
                             help='X component of plane normal')
    plane_group.add_argument('--ny', type=float, default=1.,
                             help='Y component of plane normal')
    plane_group.add_argument('--nz', type=float, default=0.,
                             help='Z component of plane normal')
    plane_group.add_argument('--plane_dist', type=float, default=0.,
                             help='Distance of plane from origin')

    options = arg_parser.parse_args()

    device = get_cuda_if_available()

    model = modules.Siren.load(options.model_path)
    model.to(device)

    plane_normal = torch.tensor([options.nx, options.ny, options.nz], device=device)
    F.normalize(plane_normal, dim=-1, out=plane_normal)
    X, Y, Z = cross_section(model, plane_normal, plane_dist=options.plane_dist, resolution=300)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    Z = Z.cpu().numpy().squeeze()

    contours_colors = plt.contourf(X, Y, Z, 13, cmap='plasma')
    contours_lines = plt.contour(X, Y, Z, 13, colors='black')
    plt.clabel(contours_lines, inline=True, fontsize=8)
    plt.colorbar(contours_colors)

    plt.show()


if __name__ == '__main__':
    main()
