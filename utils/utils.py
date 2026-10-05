"""
Various auxiliary functions
"""
import os
import time

import torch
from torch import Tensor


def runtime(func):
    """Calculate `func` running time"""

    def runtime(*args, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start = time.time()
        ret = func(*args, **kwargs)
        tm = time.time() - start
        return ret, tm

    return runtime


def fftshift(x: Tensor, dims: tuple = (-2, -1)) -> Tensor:
    """
    Correct fft-shift for many batches.
    Parameters:
        x (Tensor):
            Input tensor.
            Must have shape `(B, C, H ,W)`.
        dims (tuple):
            Dimensions to shift.
            Default is `(-2,-1)`
    Returns:
        out (Tensor):
            Shifted tensor.
    """
    shifts = [x.size(dim) // 2 for dim in dims]
    return torch.roll(x, shifts=shifts, dims=dims)


def linear_normalize(x: Tensor) -> Tensor:
    """
    Normalize input tensor via linear scaling.
    Parameters:
        x (Tensor):
            Input tensor.
    Returns:
        out (Tensor):
            Scaled tensor.
    """
    return (x - x.min()) / (x.max() - x.min())


def create_dir(path: str, verbose: bool = False) -> None:
    """
    Create new folder
    Parameters:
        path (str):
            Path to new folder.
        verbose (bool):
            Is a flag.
            If True, shows info message.
    """
    os.makedirs(path, exist_ok=True)
    if verbose:
        print(f"[info] Create new folder: {path}")


def align_bs(input: Tensor, new_bs: int) -> Tensor:
    """
    Align `input` batch size to `new_bs`
    Parameters:
        input (Tensor):
            Input tensor.
            Shape is `(B, C, H ,W)` where B is an old batch size.
        new_bs (int):
            New batch size.
    Returns:
        out (Tensor):
            Tensor with `new_bs` batch size.
    """
    single_t = input[0:1]
    return single_t.expand(new_bs, -1, -1, -1)
