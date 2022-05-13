import copy
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
import trimesh

from ensdf import brushes

from .sampling import sample_uniform_sphere, sample_uniform_disk, sample_uniform_mesh
from .sampling.sdf import SDFSampler
from .geoutils import triangle_area, normalize_point_cloud, normalize_trimesh, smoothfall, project_on_surface, tangent_grad


class DatasetBase(ABC):
    @abstractmethod
    def sample(self):
        raise NotImplemented


class PointCloudDataset(DatasetBase):
    def __init__(self, point_cloud_path, num_samples, device):
        super().__init__()

        point_cloud = np.genfromtxt(point_cloud_path)

        self.num_samples = num_samples
        self.device = device
        self.points  = torch.from_numpy(point_cloud[:, :3]).float().to(self.device)
        self.normals = torch.from_numpy(point_cloud[:, 3:]).float().to(self.device)

        # Normalize points to lie inside [-(1 - border), 1 - border]^3
        normalize_point_cloud(self.points, border = 0.15)

    def sample(self, num_samples=None):
        num_samples = num_samples or self.num_samples

        # Random indices
        rand_idcs = torch.randint(high=self.points.shape[0], size=(num_samples,), device=self.device)

        sample_points  = self.points[rand_idcs]
        sample_normals = self.normals[rand_idcs]

        sample_sdf = torch.zeros(num_samples, 1, device=self.device)

        return {
            'points': sample_points,
            'sdf': sample_sdf,
            'normals': sample_normals
        }


class MeshDataset(DatasetBase):
    def __init__(self, mesh_path, num_samples, device):
        super().__init__()

        self.num_samples = num_samples
        self.device = device

        self.mesh = trimesh.load_mesh(mesh_path)

        # Normalize mesh to lie inside [-(1 - border), 1 - border]^3
        normalize_trimesh(self.mesh, border=0.15)

        self.vertices = torch.tensor(self.mesh.vertices, dtype=torch.float32, device=device)
        # Indices must be long for pytorch
        self.triangles = torch.tensor(self.mesh.faces, dtype=torch.long, device=device)
        self.normals = torch.tensor(self.mesh.vertex_normals, dtype=torch.float32, device=device)

        self.areas = triangle_area(self.vertices[self.triangles])
        self.area_dist = torch.distributions.Categorical(self.areas)

    def sample(self, num_samples=None):
        num_samples = num_samples or self.num_samples

        sample_points, sample_normals = sample_uniform_mesh(
            self.vertices, self.triangles,
            self.normals, self.area_dist,
            num_samples
        )
        sample_sdf = torch.zeros(num_samples, 1, device=self.device)

        return {
            'points': sample_points,
            'sdf': sample_sdf,
            'normals': sample_normals
        }


class SphereSDFDataset(DatasetBase):
    def __init__(self, radius, num_samples, device):
        super().__init__()

        self.num_samples = num_samples
        self.device = device
        self.radius = radius

    def sample(self, num_samples=None):
        num_samples = num_samples or self.num_samples

        sample_normals = sample_uniform_sphere(num_samples, self.device)
        sample_points = self.radius * sample_normals
        sample_sdf = torch.zeros(num_samples, 1, device=self.device)

        return {
            'points': sample_points,
            'sdf': sample_sdf,
            'normals': sample_normals
        }


class RegularizationDataset(DatasetBase):
    def __init__(self, num_samples, device):
        super().__init__()

        self.num_samples = num_samples
        self.device = device

    def sample(self, num_samples=None):
        num_samples = num_samples or self.num_samples
        sample_points = 2.3 * torch.rand(self.num_samples, 3, device=self.device) - 1.15

        return {'points': sample_points}


class SDFEditingDataset(DatasetBase):
    def __init__(
        self, model, brush : brushes.BrushBase, device,
        sample_model_update_iters,
        num_interaction_samples,
        num_model_samples
    ):
        super().__init__()

        self.model = model
        self.device = device
        self.sample_model_update_iters = sample_model_update_iters
        self.iters = 0

        self.brush = brush

        self.num_interaction_samples = num_interaction_samples
        self.num_model_samples = num_model_samples

        self.sdf_sampler = SDFSampler(
            copy.deepcopy(self.model),
            self.device, self.num_model_samples
        )
        
        self.model_samples = None
        self.next_model_samples = next(self.sdf_sampler)

    def sample(self):
        self.model_samples = self.next_model_samples

        self.iters += 1
        if self.iters % self.sample_model_update_iters == 0:
            self.sdf_sampler.model = copy.deepcopy(self.model)

        self.next_model_samples = next(self.sdf_sampler)

        # Model samples
        keep_cond = ~self.brush.inside_interaction(self.model_samples['points'])
        filtered_model_points  = self.model_samples['points'][keep_cond]
        filtered_model_normals = self.model_samples['normals'][keep_cond]
        filtered_model_sdf     = self.model_samples['sdf'][keep_cond]

        # self.num_interaction_samples = self.num_model_samples - filtered_model_points.shape[0]

        # Interaction samples
        inter_points, inter_sdf, inter_normals = (
            self.brush.sample_interaction(self.model, self.num_interaction_samples)
        )

        # Collect all samples
        sample_points  = torch.cat((filtered_model_points, inter_points))
        sample_normals = torch.cat((filtered_model_normals, inter_normals))
        sample_sdf     = torch.cat((filtered_model_sdf, inter_sdf))

        return {
            'points': sample_points,
            'sdf': sample_sdf,
            'normals': sample_normals
        }
