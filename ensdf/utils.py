import torch


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
