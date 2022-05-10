import torch
import torch.nn.functional as F


def phong_shading(
    points, normals,
    light_pos, light_colors,
    eye_pos,
    ks=torch.tensor([0.3, 0.3, 0.3]),
    kd=torch.tensor([0.2, 0.2, 0.7]),
    ka=torch.tensor([0.1, 0.1, 0.1]),
    shininess = 8
):
    device = points.device

    points_v = points.view(-1, 1, 3)
    normals_v = normals.view(-1, 1, 3)
    
    light_dirs = light_pos.view(1, -1, 3) - points_v
    F.normalize(light_dirs, dim=-1, out=light_dirs)
    
    color = torch.zeros(points_v.shape[0], 3, device=device)

    # Diffuse
    n_comp = (light_dirs * normals_v).sum(-1, keepdim=True)
    diffuse = F.relu(n_comp)
    diffuse = diffuse * light_colors.view(1, -1, 3)
    diffuse = diffuse.sum(1)
    kd = kd.view(1, 3).to(device)
    diffuse.mul_(kd)
    
    color.add_(diffuse)
    
    # Specular
    eye_dirs = eye_pos.view(1, 1, 3) - points_v
    F.normalize(eye_dirs, dim=-1, out=eye_dirs)
    t_comp = light_dirs - n_comp * normals_v
    reflected_light_dirs = torch.sub(light_dirs, t_comp, alpha=2, out=light_dirs)
    specular = (reflected_light_dirs * eye_dirs).sum(-1, keepdim=True)
    torch.pow(specular, shininess, out=specular)
    specular = specular * light_colors.view(1, -1, 3)
    torch.lerp(torch.tensor(0., device=device), specular, F.relu(n_comp), out=specular)
    # specular.mul_(torch.heaviside(n_comp, torch.tensor(0., device=device), out=n_comp).view(-1, 1))
    specular = specular.sum(1)
    ks = ks.view(1, 3).to(device)
    specular.mul_(ks)

    color.add_(specular)

    # Ambient
    color.add_(ka.view(1, 3).to(device))

    color = color.reshape_as(points)

    return color

def shade_normals(normals):
    return normals * 0.5 + 0.5
