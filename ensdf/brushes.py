from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from .sampling import sample_uniform_disk
from .geoutils import project_on_surface, tangent_grad, smoothfall


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
        return (points - self.inter_point).norm(dim=1) <= self.radius
    
    @abstractmethod
    def sample_interaction(self, model, num_samples):
        raise NotImplemented


class SimpleBrush(BrushBase):
    def sample_interaction(self, model, num_samples):
        disk_samples = sample_uniform_disk(self.inter_normal, num_samples)
        disk_samples = disk_samples.squeeze(0) * self.radius
        disk_sample_norms = torch.norm(disk_samples, dim=-1)

        y, dy = smoothfall(disk_sample_norms, radius=self.radius, return_derivative=True)
        y, dy = self.intensity * y.unsqueeze(1), self.intensity * dy.unsqueeze(1)

        normals = F.normalize(disk_samples, dim=-1)
        normals.mul_(dy)

        points = torch.add(disk_samples, self.inter_point, out=disk_samples)
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

        tan_grad = tangent_grad(projected_normals, self.inter_normal)
        normals.add_(tan_grad)
        torch.add(self.inter_normal, normals, alpha=-1, out=normals)
        F.normalize(normals, dim=-1, out=normals)

        sdf = torch.zeros(num_samples, 1, device=self.inter_point.device)

        return points, sdf, normals
