from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F


def euler_to_matrix(yaw, pitch, roll):
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)

    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)

    c_roll = np.cos(roll)
    s_roll = np.sin(roll)    

    yaw_mat = np.array(
        [
            [c_yaw , 0., s_yaw],
            [0.    , 1., 0.   ],
            [-s_yaw, 0., c_yaw]
        ]
    )
    pitch_mat = np.array(
        [
            [1., 0.     , 0.      ],
            [0., c_pitch, -s_pitch],
            [0., s_pitch, c_pitch ]
        ]
    )
    roll_mat = np.array(
        [
            [c_roll, -s_roll, 0.],
            [s_roll, c_roll , 0.],
            [0.    , 0.     , 1.]
        ]
    )

    return yaw_mat @ pitch_mat @ roll_mat


def spherical_to_cartesian(phi, theta, radius):
    # y is up
    y = radius * np.cos(theta)
    x = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi) * np.sin(theta)

    return x, y, z


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
