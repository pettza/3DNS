import numpy as np
import open3d as o3d
import trimesh


def ball_pivoting(points, normals, radii=[0.05, 0.1, 0.2, 0.4]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    return trimesh.Trimesh(
        vertices = np.asarray(o3d_mesh.vertices),
        faces    = np.asarray(o3d_mesh.triangles),
        normals  = np.asarray(o3d_mesh.vertex_normals)
    )