import torch


def cross_section(model, plane_normal, plane_dist, resolution):
    device = plane_normal.device

    normal_abs = torch.abs(plane_normal)
    if normal_abs[0] > normal_abs[1]:
        x = plane_normal[0]
        z = plane_normal[2]
        tan_axis = torch.tensor([-z, 0.0, x], device=device) / torch.sqrt(z**2 + x**2)
    else:
        y = plane_normal[1]
        z = plane_normal[2]
        tan_axis = torch.tensor([0.0, z, -y], device=device) / torch.sqrt(z**2 + y**2)
    cotan_axis = torch.cross(plane_normal, tan_axis)

    res = torch.linspace(-1, 1, resolution, device=device)
    X, Y = torch.meshgrid(res, res, indexing='xy')
    points =  tan_axis * X.unsqueeze(-1) + cotan_axis * Y.unsqueeze(-1) + plane_dist * plane_normal

    with torch.no_grad():
        Z = model(points)

    return X, Y, Z
