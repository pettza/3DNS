from os import TMP_MAX
import torch


class AABB:
    def __init__(self, center, half_extent, device='cpu'):
        self.center = torch.tensor(center, device=device)
        self.half_extent = torch.tensor(half_extent, device=device)
    
    def contains(self, points):
        t = points - self.center
        torch.abs(t, out=t)
        return (t < self.half_extent).all(dim=-1, keepdim=True)

    def intersect(self, origins, directions):
        inv_dir = 1. / directions
        t0 = (self.center - self.half_extent - origins) * inv_dir
        t1 = (self.center + self.half_extent - origins) * inv_dir
        tmin, min_indices = torch.minimum(t0, t1).max(dim=-1, keepdim=True)
        tmax, max_indices = torch.maximum(t0, t1).min(dim=-1, keepdim=True)

        hit = tmin <= tmax
        thit = torch.relu(torch.where(tmin >= 0., tmin, tmax))

        return hit, thit

    def volume(self):
        return torch.prod(2 * self.half_extent)