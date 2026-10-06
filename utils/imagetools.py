"""
Functions to visualize images/losses
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import cv2


def plot_images(
    images: list,
    titles: list | None = None,
    rows: int = 1,
    show_range: bool = True,
    figsize: tuple[int, int] = (12, 8),
    fontsize: int = 10,
    save_path: str | None = None,
) -> None:
    """
    Plot several images.
    Parameters:
        images (list):
            List of images.
            Can be torch tensors or numpy arrays.
        titles (list):
            List of titles (correspondance to `images`).
            If None, images will not have titles.
            Default is None.
        rows (int):
            Number of rows.
            Default is `1`.
        figsize (tuple):
            Figure size.
            Default is `(12, 8)`.
        save_path (str):
            Path to save output image
            If None, image will not be saved.
            Default is None.
    """
    cols = len(images) // rows
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.atleast_1d(axs).ravel()

    for i, img in enumerate(images):
        # convert to numpy array
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().to(torch.float32).numpy()
        elif isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
        else:
            arr = np.array(img, dtype=np.float32)

        # reduce channels
        arr = arr.squeeze()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = arr.transpose(1, 2, 0)

        axs[i].imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=1)
        axs[i].axis("off")
        if titles:
            if show_range:
                axs[i].set_title(
                    f"{titles[i]}\nmin={arr.min():.3f}, max={arr.max():.3f}",
                    fontsize=fontsize,
                )
            else:
                axs[i].set_title(titles[i], fontsize=fontsize)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()


def add_psf_on_image(
    img: torch.Tensor, psf: torch.Tensor, part_size: tuple[int, int] = (100, 100)
) -> torch.Tensor:
    """
    Add small PSF fragment on image.
    Parameters:
        img (Tensor):
            Source image.
        psf (Tensor):
            Source PSF.
        part_size (tuple):
            Size of PSF fragment on image.
            Default is `(100, 100)`.
    Returns:
        out (Tensor):
            Output images with PSF fragments.
    """
    # clone source
    img_copy = img.clone().detach()
    psf_copy = psf.clone().detach()

    # centering & normalizing PSF
    for i in range(psf_copy.shape[0]):
        psf_copy[i] = torch.fft.fftshift(psf_copy[i]) / psf_copy[i].max()

    # locate fragment
    h, w = psf_copy.shape[2], psf_copy.shape[3]
    f_h, f_w = part_size
    x_, y_ = int((h - f_h) / 2), int((w - f_w) / 2)

    # add fragment on image
    fragment = psf_copy[:, :, x_ : x_ + f_h, y_ : y_ + f_w]
    img_copy[:, :, :f_h, w - f_w :] = fragment

    return img_copy


class Rectangle:
    def __init__(
        self, x: int = 0, y: int = 0, size: int = 1, color: list[int] = (0, 0, 0)
    ) -> None:
        """Class constructor"""

        self.x = x
        self.y = y
        self.size = size
        self.color = color

    def up_left(self, numpy: bool = True) -> list[int]:
        """Return ccordinates of upper left corner"""
        if numpy:
            return (self.y, self.x)
        return (self.x, self.y)

    def low_right(self, numpy: bool = True) -> list[int]:
        """Return ccordinates of lower right corner"""
        if numpy:
            return (self.y + self.size, self.x + self.size)
        return (self.x + self.size, self.y + self.size)


def plot_histogram(array, savepath="results/images/hist.png"):
    array = np.array(array)

    plt.figure()
    hist, bins = np.histogram(
        array, bins=100, density=False, range=(array.min(), array.max())
    )
    plt.plot(bins[:-1], hist, color="b", linewidth=2)
    plt.grid(True)

    # Убираем оси и рамку
    # plt.axis('off')          # полностью отключает оси, подписи, рамку
    # Альтернативно, можно оставить оси но без меток:
    # plt.xticks([])
    # plt.yticks([])

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def save_histogram(torch_img, save_path="results/images/hist.png", figsize=(6, 4)):
    # Берём первый элемент батча, (C, H, W) -> (H, W, C)
    img_np = torch_img[0].cpu().detach().numpy().transpose(1, 2, 0)

    # Создаём фигуру с белым фоном, axes на всю площадь
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])  # без полей
    fig.add_axes(ax)
    ax.set_facecolor("white")

    # Строим гистограммы
    colors = ("Red", "Green", "Blue")
    for i, color in enumerate(colors):
        hist, bins = np.histogram(
            img_np[:, :, i].ravel(), bins=100, density=False, range=(0, 1)
        )  # принудительно от 0 до 1
        ax.plot(bins[:-1], hist, color=color.lower(), label=color, linewidth=2)

    # Настройка осей: оставляем только нижнюю (X) от 0 до 1
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    # Убираем тики и метки с левой оси
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Настройка оси X от 0 до 1
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    # Убираем внутренние отступы данных
    ax.margins(0)

    # Легенда (опционально, можно убрать)
    # ax.legend(loc='upper right', frameon=False)

    # Сохраняем без полей, с белым фоном
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close()


def save_image(
    src: torch.Tensor, title: str, rectangles: list[Rectangle] | None = None
) -> None:
    "Save tensor as image"
    src_np = src[0].permute(1, 2, 0).cpu().detach().numpy()
    src_np = (255 * src_np).astype(np.uint8)
    src_np = cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR)
    if rectangles:
        for rect in rectangles:
            cv2.rectangle(
                src_np, rect.up_left(), rect.low_right(), rect.color, thickness=2
            )
    cv2.imwrite(title, src_np)


def plot_loss(
    read_path: str = "logs/metrics.json",
    save_path: str = "logs/loss_plot.png",
    figsize: tuple[int, int] = (12, 8),
    log_scale: bool = True,
):
    """
    Plot train/val loss using information from json.
    Parameters:
        read_path (str):
            Path to json file with train/val data.
            Default is `"logs/metrics.json"`.
        read_path (str):
            Path to save graphics.
            Default is `"logs/loss_plot.png"`.
        figsize (tuple):
            Figure size.
            Default is `(12, 8)`.
        log_scale (bool):
            Is a flag.
            If True, show losses in log scale.
            Default is True.
    """
    with open(read_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = data["metrics"]["epoch"]
    train_loss = data["metrics"]["train_loss"]
    val_loss = data["metrics"]["val_loss"]

    plt.figure(figsize=figsize)

    if log_scale:
        train_loss = np.log(train_loss)
        val_loss = np.log(val_loss)

    plt.plot(
        epochs,
        train_loss,
        "b-",
        label="Train Loss",
        marker="o",
        linewidth=2,
        markersize=6,
    )
    plt.plot(
        epochs,
        val_loss,
        "r-",
        label="Validation Loss",
        marker="s",
        linewidth=2,
        markersize=6,
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True)

    plt.xticks(epochs)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tone_mapping(
    model, span_prms=(-2, 3, 1e-2), savepath="logs/tm_validation.png"
):
    span = np.arange(*span_prms)
    outs = {"mean": [], "spread": []}

    for l in span:
        inp = l * torch.ones((1, 3, 512, 512)).cuda()
        out = model(inp)

        outs["mean"].append(out.mean().item())
        outs["spread"].append(out.std().item())

    plt.figure(figsize=(12, 6))

    plt.plot(span, outs["mean"], "r-", label="Mapped brightness", linewidth=2)
    plt.plot(span, outs["spread"], "b-", label="Sustain", linewidth=2)

    plt.xlabel("Input brightness", fontsize=12)
    plt.ylabel("Output brightness", fontsize=12)
    plt.title("Tone Mapping Function", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()
