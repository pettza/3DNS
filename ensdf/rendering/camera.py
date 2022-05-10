from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from ..geoutils import euler_to_matrix, spherical_to_cartesian


class CameraBase(ABC):
    def __init__(self, fov, resolution):
        self.fov = fov
        self.resolution = resolution
  
    @property
    @abstractmethod
    def position(self):
        raise NotImplemented
    
    @property
    @abstractmethod
    def orientation_matrix(self):
        raise NotImplemented

    def generate_rays(self):
        x_half_extent = np.tan(self.fov / 2)
        x_half_extent -= x_half_extent / self.resolution[0]
        y_half_extent = self.resolution[1] / self.resolution[0] * x_half_extent

        directions = np.swapaxes(
            np.mgrid[
                x_half_extent:-x_half_extent:self.resolution[0] * 1j,
                -y_half_extent:y_half_extent:self.resolution[1] * 1j,
                1:2
            ],
            0, -1
        ).squeeze()
        directions = torch.tensor(directions, dtype=torch.float)
        directions = directions.view(-1, 3, 1)
        directions = torch.matmul(self.orientation_matrix.view(1, 3, 3), directions)
        directions = directions.view(self.resolution[0], self.resolution[1], 3)
        F.normalize(directions, dim=-1, out=directions)

        origins = (
            self.position
                .repeat(self.resolution[0] * self.resolution[1], 1)
                .reshape(self.resolution[0], self.resolution[1], 3)
        )

        return origins, directions


class SimpleCamera(CameraBase):
    def __init__(self, fov, resolution, position, yaw, pitch, roll):
        super().__init__(fov, resolution)
        
        self.position = torch.tensor(position, dtype=torch.float)

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    @property
    def position(self):
        return self.position
    
    @property
    def orientation_matrix(self):
        return torch.tensor(euler_to_matrix(self.yaw, self.pitch, self.roll), dtype=torch.float)


class OrbitingCamera(CameraBase):
    def __init__(self, fov, resolution, phi, theta, radius):
        super().__init__(fov, resolution)

        self.phi = phi
        self.theta = theta
        self.radius = radius

    @property
    def position(self):
        return torch.tensor(spherical_to_cartesian(self.phi, self.theta, self.radius), dtype=torch.float)
    
    @property
    def orientation_matrix(self):
        return torch.tensor(euler_to_matrix(self.phi + np.pi, -self.theta + np.pi/2, 0.), dtype=torch.float)
