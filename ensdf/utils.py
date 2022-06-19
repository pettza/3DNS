import torch
import numpy as np


def get_cuda_if_available():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def div_round_up(dividend, divisor):
    return (dividend + divisor - 1) // divisor


def batch_computation(*in_tensors, fn, collect_fn, batch_size):
    """
    Does a computation in batches

    Args:
        *in_tensors (torch.Tesnor): Tensors that will be split along the 0 dimension and
                                    passed to fn as positional arguments
        fn (Callable): The function that will process the batches
        collect_fn (Callable): The function that will take the results of fn as positional arguments
        batch_size (int): The size of batches (except perhaps the last one)
    """
    for batched_in_tensors in zip(*[torch.split(t, batch_size) for t in in_tensors]):
        batched_out = fn(*batched_in_tensors)
        collect_fn(*batched_out)


def binary_search(low, high, target_value, fn):
    """
    Finds the first object in the the range [low, hish] in an indexed colletion 
    whose value is >= to the target_value. The value needs to be monotonically 
    increasing (not strictly) with the index.

    Args:
        low (int): The first index in the range of the search
        high (int): The first index in the range of the search
        target_value: The value that we want to look for
        fn (Callable): A callable that returns the object and its value given its index

    Returns:
        tuple: A 3-tuple containing the found index, the object at that index and its value
    """
    while low < high:
        mid = (low + high) // 2
        mid_obj, mid_value = fn(mid)

        if mid_value < target_value:
            low = mid + 1
        else:
            high = mid
    
    low_obj, low_value = fn(low)
    return low, low_obj, low_value


def size_of_trimesh(mesh):
    size_of_float = 4
    size_of_int = 4

    size_of_vertices = np.prod(mesh.vertices.shape) * size_of_float
    size_of_normals = size_of_vertices
    size_of_indices = np.prod(mesh.faces.shape) * size_of_int

    total_size = size_of_vertices + size_of_normals + size_of_indices

    return total_size


def simplify_trimesh(mesh, target_size):
    def f(num_faces):
        dec_mesh = mesh.simplify_quadratic_decimation(num_faces)
        dec_mesh_size = size_of_trimesh(dec_mesh)
        return dec_mesh, dec_mesh_size

    return binary_search(1, mesh.faces.shape[0], target_size, f)
        