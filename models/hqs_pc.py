"""
Main HQS algorithm
"""

import torch
import torch.nn as nn
import torch.fft as fft
from tqdm import trange

from torch import Tensor

from typing import NamedTuple, List, Literal

from .blocks import W_solver, Z_solver

from utils.filters import grad, sobel
from utils.loss import PreCompensationLoss, align_bs


def init_var(
    source: Tensor, 
    init_type: Literal["source", "zeros", "ones", "random"]
) -> Tensor:
    """Initialize variable"""
    if init_type == "source":
        return source.clone()
    elif init_type == "zeros":
        return torch.zeros_like(source)
    elif init_type == "ones":
        return torch.ones_like(source)
    elif init_type == "random":
        return torch.randn_like(source)
    else:
        raise ValueError(f"Unexpected initial variable type: {init_type}")


class HQS_PC_Parameters(NamedTuple):
    """Classic HQS-PC parameters"""

    # optimization prms
    max_iter: int = 1000
    epsilon: float | None = 1e-4
    z0_type: Literal["source", "zeros", "ones", "random"] = "source"
    verbose: bool = False
    # loss prms
    filters: Tensor | Literal["grad", "sobel"] = "grad"
    theta: float = 1e-6
    r_norm_type: Literal["l1", "l2"] = "l1"
    # parameters
    beta: float | List[float] = 1e-3
    gamma: float = 1e-6


def hqs_pc(
    t: Tensor, k: Tensor, parameters: HQS_PC_Parameters = HQS_PC_Parameters()
) -> torch.Tensor:
    """Compute HQS-PC for image `t` and kernel `k`"""
    batch_size = t.size(0)

    # prepare solvers
    w_solver = W_solver(parameters.r_norm_type)
    z_solver = Z_solver()

    # prepare parameters
    beta_tmp = parameters.beta
    if isinstance(beta_tmp, float):
        beta = beta_tmp * torch.ones(batch_size, parameters.max_iter, device=t.device)
    else:
        beta = torch.tensor(beta_tmp, device=t.device)

    gamma = parameters.gamma * torch.ones(batch_size, device=t.device)
    eps = parameters.epsilon

    filter_tmp = parameters.filters
    if isinstance(filter_tmp, str):
        if filter_tmp == "grad":
            filters = grad(t.shape).to(t.device)
        elif filter_tmp == "sobel":
            filters = sobel(t.shape).to(t.device)
        else:
            raise ValueError(f"Unexpected filter type: {filter_tmp}")
    else:
        filters = filter_tmp

    # fft of blur kernel
    kernel_fft = fft.fft2(k)

    # fft of filters
    filters_fft = fft.fft2(filters)

    # fft of product K^T*t
    kt_fft = torch.conj(kernel_fft) * fft.fft2(t)

    # initial z value
    z0 = init_var(t, parameters.z0_type)

    # iteration loop
    loss = PreCompensationLoss(
        theta=parameters.theta,
        filters=filters,
        regularization=parameters.r_norm_type,
        reduction="mean",
    )
    loss_step = []
    for i in (
        trange(parameters.max_iter)
        if parameters.verbose
        else range(parameters.max_iter)
    ):
        w = w_solver(z0, filters_fft, parameters.theta / gamma)
        p = torch.clip(z0, 0, 1)
        z0 = z_solver(p, kt_fft, kernel_fft, w, filters_fft, gamma, beta[i])
        # loss
        loss_step.append(loss(p, t, k).item())
        # stop criterion
        p_new = torch.clip(z0, 0, 1)
        if eps and torch.mean((p - p_new).abs()) <= eps:
            return p_new, loss_step, i

    return p_new, loss_step, parameters.max_iter


class HQS_PC(nn.Module):
    """HQS model for image PreCompensation"""

    def __init__(
        self,
        n_iter: int,
        filters: Tensor,
        theta: float = 1e-6,
        regularization: Literal["l2", "l1", "l0"] = "l1",
        init_type: Literal["source", "zeros", "ones", "random"] = "source",
        init_beta: float = 1.0
    ) -> None:
        """Class comstructor"""
        super(HQS_PC, self).__init__()

        self.n_iter = n_iter
        self.init_type = init_type

        self.filters = filters
        self.theta = theta
        self.reg = regularization

        # training parameters
        self.beta  = nn.Parameter(init_beta * torch.ones(n_iter))
        self.gamma = nn.Parameter(torch.ones(n_iter))

        self.w_solver = W_solver(regularization)
        self.z_solver = Z_solver()

    def clip_weights(self) -> None:
        """Clip weights to avoid negative values"""
        self.beta.data.clamp_(1e-8)
        self.gamma.data.clamp_(1e-8)

    def forward(self, t: Tensor, k: Tensor, p0: Tensor | None = None) -> Tensor:
        """Compute HQS-PC for image `t` and kernel `k`"""
        # align batch sizes
        self.filters = align_bs(self.filters, t.size(0))

        # fft of blur kernel
        kernel_fft = fft.fft2(k)

        # fft of filters
        filters_fft = fft.fft2(self.filters)

        # fft of product K^T*t
        kt_fft = torch.conj(kernel_fft) * fft.fft2(t)

        # initial variables
        p = init_var(t, self.init_type) if p0 is None else p0
        d_fft = filters_fft.unsqueeze(1)
        p_fft = fft.fft2(p).unsqueeze(2)
        w = fft.ifft2(d_fft * p_fft).real

        # iteration loop
        for i in range(self.n_iter):
            # parameters
            beta  = self.beta[i]
            gamma = self.beta[i]  # self.gamma[i]

            # consistent optimization
            z = self.z_solver(p, kt_fft, kernel_fft, w, filters_fft, gamma, beta)
            w = self.w_solver(z, filters_fft, self.theta / gamma)
            p = torch.clip(z, 0, 1)

        return p