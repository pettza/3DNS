from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import trimesh

from .sampling import sample_uniform_disk
from .geoutils import project_on_surface, tangent_grad, intersect_plane, \
                      linear_fall, cubic_fall, quintic_fall
from .diff_operators import gradient


class BrushBase(ABC):
    def __init__(self, radius=0.07, intensity=0.04):
        self.radius = radius
        self.intensity = intensity
        self.inter_point = self.inter_normal = self.inter_sdf = None
    
    def set_interaction(self, inter_point, inter_normal, inter_sdf):
        self.inter_point  = inter_point
        self.inter_normal = F.normalize(inter_normal, dim=-1)
        self.inter_sdf    = inter_sdf

    def inside_interaction(self, points):
        return (points - self.inter_point).norm(dim=-1) <= self.radius

    @abstractmethod
    def sample_interaction(self, model, num_samples):
        raise NotImplemented


class SimpleBrush(BrushBase):
    def __init__(self, brush_type, **kwargs):
        super().__init__(**kwargs)
        self.brush_type = brush_type

    @property
    def brush_type(self):
        return self._brush_type

    @brush_type.setter
    def brush_type(self, brush_type):
        self._brush_type = brush_type

        if brush_type == 'linear':
            self.template = linear_fall
        elif brush_type == 'cubic':
            self.template = cubic_fall
        elif brush_type == 'quintic':
            self.template = quintic_fall
        else:
            raise ValueError(f'{brush_type} is not a valid type for {type(self).__name__}')
    
    def evaluate_template_on_tangent_disk(self, disk_points):
        disk_sample_norms = torch.norm(disk_points, dim=-1)
        disk_sample_norms.requires_grad = True
        y = self.intensity * self.template(disk_sample_norms, radius=self.radius)
        dy = gradient(y, disk_sample_norms)
        y, dy = y.detach().unsqueeze(-1), dy.detach().unsqueeze(-1)

        return y, dy

    def adjust_normals(self, projected_normals, disk_points, template_grad):
        normals = F.normalize(disk_points, dim=-1)
        normals.mul_(template_grad)
        tan_grad = tangent_grad(projected_normals, self.inter_normal)
        normals.add_(tan_grad)
        torch.add(self.inter_normal, normals, alpha=-1, out=normals)
        F.normalize(normals, dim=-1, out=normals)
        return normals

    def sample_interaction(self, model, num_samples):
        disk_samples = sample_uniform_disk(self.inter_normal, num_samples)
        disk_samples = disk_samples.squeeze(0) * self.radius
        
        y, dy = self.evaluate_template_on_tangent_disk(disk_samples)

        points = torch.add(disk_samples, self.inter_point)
        projected_points, projected_sdf, projected_normals = project_on_surface(
            model,
            points,
            num_steps=2
        )
        points = torch.addcmul(
            projected_points,
            y,
            self.inter_normal,
            out=projected_points
        )

        normals = self.adjust_normals(projected_normals, disk_samples, dy)

        sdf = torch.zeros(num_samples, 1, device=self.inter_point.device)

        return points, sdf, normals

    def deform_mesh(self, mesh):
        mesh_points = torch.tensor(mesh.vertices, dtype=torch.float32)
        mesh_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)

        inside = self.inside_interaction(mesh_points)
        points_in_interaction = mesh_points[inside]
        normals_in_interaction = mesh_normals[inside]

        plane_points = intersect_plane(self.inter_normal, self.inter_point, points_in_interaction, normals_in_interaction)
        plane_points -= self.inter_point

        y, dy = self.evaluate_template_on_tangent_disk(plane_points)

        adjusted_normals = self.adjust_normals(normals_in_interaction, plane_points, dy)
        
        mesh_points[inside] = points_in_interaction + y * self.inter_normal
        mesh_normals[inside] = adjusted_normals

        deformed_mesh = trimesh.Trimesh(
            vertices=mesh_points,
            vertex_normals=mesh_normals,
            faces=mesh.faces
        )

        return deformed_mesh
