import torch
import numpy as np
from math import inf

from ..utils import batch_computation
from ..diff_operators import gradient


RAYMARCH_CONVERGENCE_THRESHOLD = 0.001


def raymarch(model, aabb, origins, directions, batch_size=1<<22, num_iter=40):
    model.eval()

    device = origins.device

    batch_dims = origins.shape[:-1]
    batch_dims_size = np.product(batch_dims)

    points = torch.empty(batch_dims_size, 3, device=device)
    normals = torch.empty(batch_dims_size, 3, device=device)
    sdf = torch.empty(batch_dims_size, 1, device=device)
    ray_hit = torch.empty(batch_dims_size, 1, device=device, dtype=torch.bool)

    batch_counter = 0

    def fn(origins, directions):
        # Move rays outside the bounding box to the points they intersect
        hit, thit = aabb.intersect(origins, directions)
        inside_bb = aabb.contains(origins)
        t = torch.where(hit & ~inside_bb, thit, torch.tensor(0., device=origins.device))
        b_points = origins + t * directions

        with torch.no_grad():
            for i in range(num_iter):
                b_sdf = model(b_points)
                b_points.addcmul_(b_sdf, directions)
            
        b_points = b_points.detach()
        b_points.requires_grad = True
        
        b_sdf = model(b_points)
        
        b_normals = gradient(b_sdf, b_points).detach()
        b_points = b_points.detach()
        
        inside_bb = aabb.contains(b_points)
        b_ray_hit = inside_bb # & (b_sdf < RAYMARCH_CONVERGENCE_THRESHOLD)
        
        return b_points, b_normals, b_sdf, b_ray_hit

    def collect_fn(b_points, b_normals, b_sdf, b_ray_hit):
        nonlocal batch_counter, points, normals, sdf, ray_hit

        curr_batch_size = b_points.shape[0]
        
        points[batch_counter:batch_counter+curr_batch_size] = b_points
        normals[batch_counter:batch_counter+curr_batch_size] = b_normals
        sdf[batch_counter:batch_counter+curr_batch_size] = b_sdf
        ray_hit[batch_counter:batch_counter+curr_batch_size] = b_ray_hit

        batch_counter += curr_batch_size

    batch_computation(origins.view(-1, 3), directions.view(-1, 3), fn=fn, collect_fn=collect_fn, batch_size=batch_size)

    points = points.view(*batch_dims, 3)
    normals = normals.view(*batch_dims, 3)
    sdf = sdf.view(*batch_dims, 1)
    ray_hit = ray_hit.view(*batch_dims, 1)

    return points, normals, sdf, ray_hit


def raymarch_single_ray(model, aabb, origin, direction, max_iter=40):
    model.eval()
    
    # Move raya outside the bounding box to the points they intersect
    hit, thit = aabb.intersect(origin, direction)
    inside_bb = aabb.contains(origin)
    point = origin + thit.where(hit & ~inside_bb, 0.) * direction
    
    with torch.no_grad():
        for i in range(max_iter):
            sdf = model(origin)
            torch.addcmul(point, sdf, direction, out=origin)

            ray_hit = sdf < RAYMARCH_CONVERGENCE_THRESHOLD
            if ray_hit:
                break
    
    point = point.detach()
    point.requires_grad = True
    
    sdf = model(point)

    normal = gradient(sdf, point).detach()
    point = point.detach()
    sdf = sdf.detach()

    return point, normal, sdf, ray_hit
