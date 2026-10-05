"""
Convolution tools.
Using Fast Fourier Transform.
"""

import torch
from torch import Tensor
import torch.nn.functional as F

import torch.fft as fft

# from olimp.processing import fft_conv as pyolimp_conv
from olimp.processing import resize_kernel

from .utils import fftshift


def conv(x: Tensor, y: Tensor, spatial: bool = True) -> Tensor:
    """
    FFT base method for fast convolution.
    Input types are expected to be `float32` and size must be equal.

    **Note**: if x have C > 1 channels and y have N > 1 channels,
    output will have `(B, C, N, H, W)` shape.

    Parameters:
        x (Tensor):
            First tensor.
            Must have shape `(B, C, H, W)`.
        y (Tensor):
            Second tensor.
            Must have shape `(B, N, H, W)`.
        spatial (bool):
            If `True`, considers that tensors given in spatial domain.
            Default is `True`.

    Returns:
        out (Tensor):
            Convolution result with shape `(B, C, N, H, W)`.
            If N=1 (PSF case) shape is `(B, C, H, W)`.
    """
    assert x.dtype == torch.float32, x.dtype
    assert y.dtype == torch.float32, y.dtype
    assert x.shape[-2:] == y.shape[-2:], (
        "Expected equal shapes, got: " f"x={x.shape[-2:]}, y={y.shape[-2:]}"
    )

    if spatial:
        # make fourier if tensors in spatial domain
        x = fft.fft2(x)
        y = fft.fft2(y)

    if x.shape[1] > 1 and y.shape[1] > 1:
        y = y.unsqueeze(1)
        x = x.unsqueeze(2)

    return fft.ifft2(x * y).real


def padding(input: Tensor, alpha: float = 0.03, mode: str = "replicate") -> Tensor:
    """
    Pad input tensor.
    Parameters:
        input (Tensor):
            Input tensor
        alpha (float):
            The proportion of size increase.
            Must be in range from 0 to 1.
            Default is `0.03`.
        mode (str):
            Padding type.
            Can be: `constant` (pads to 0), `reflect`, `replicate`, `circular`.
            Default is `"replicate"`.
    Returns:
        out (Tensor):
            Output padded tensor with shape H' = (1 + alpha) * H.
    """
    assert (
        alpha >= 0.0
    ) and alpha <= 1.0, f"Percent must be in range [0, 1], got: alpha={alpha:0.4f}"
    _, _, h, w = input.shape

    # additional sizes
    pad_h = int(h * alpha)
    pad_w = int(w * alpha)

    padding = (pad_w, pad_w, pad_h, pad_h)
    return F.pad(input, padding, mode=mode)


def cropping(
    image: Tensor, original_shape: tuple[int, int], alpha: float = 0.03
) -> Tensor:
    """
    Crop input tensor after `padding`.
    Parameters:
        input (Tensor):
            Input tensor.
        original_shape (tuple):
            Original tensor shape (before `padding`).
        alpha (float):
            The proportion of size increase (see `padding`).
            Must be in range from 0 to 1.
            Default is `0.03`.
    Returns:
        out (Tensor):
            Cropped tensor to original size.
    """
    h, w = original_shape

    pad_h = int(h * alpha)
    pad_w = int(w * alpha)

    return image[:, :, pad_h : pad_h + h, pad_w : pad_w + w]


def fft_conv(
    image: Tensor, psf: Tensor, alpha: float = 0.03, mode: str = "replicate"
) -> Tensor:
    """
    Convolution with padding to avoid board effects
    Parameters:
        image (Tensor):
            Source image with shape (B, C, H, W).
        psf (Tensor):
            Source PSF with shape (B, 1, H, W).
        alpha (float):
            Boundary expansion parameter (see `padding`).
            Must be in range from 0 to 1.
            Default is `0.03`.
        mode (str):
            Padding type.
            Can be: `constant` (pads to 0), `reflect`, `replicate`, `circular`.
            Default is `"replicate"`.
    """
    _, _, h, w = image.shape

    # Увеличиваем ядро
    pad_psf = padding(fftshift(psf), alpha, "constant")
    pad_psf = fftshift(pad_psf)
    pad_image = padding(image, alpha, mode)

    # Свертка расширенного ядра и паддинга
    convolved = conv(pad_image, pad_psf)

    # Обрезка до нужного размера
    return cropping(convolved, (h, w), alpha)


def decrease_kernel(psf: Tensor, factor: float = 0.5) -> Tensor:
    """
    Decrease kernel size in ``factor`` times.
    Parameters:
        psf (Tensor):
            Input PSF tensor
        factor (float):
            Decrease scale.
            Default is `0.5`.
    """

    b, _, h, w = psf.shape
    new_h, new_w = int(h * factor), int(w * factor)

    psf_resized = resize_kernel(psf, (new_h, new_w))
    psf_padded = torch.zeros(psf.shape, dtype=torch.float32, device=psf.device)

    x = (h - new_h) // 2
    y = (w - new_w) // 2
    psf_padded[:, :, x : x + new_h, y : y + new_w] = fft.fftshift(psf_resized)

    for i in range(b):
        psf_padded[i] = fft.fftshift(psf_padded[i]) / psf_padded[i].sum()
    return psf_padded
