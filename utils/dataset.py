"""
Dataset tools, including main dataset class
"""

import numpy as np
import cv2
import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms


class PreCompensationDataset(Dataset):
    """
    Dataset to image precompensation training/validation.
    """

    def __init__(
        self,
        psf_dirs: list,
        img_dirs: list,
        n_images_per_psf: int = None,
        seed: int = 42,
        gray: bool = False,
        imagenet: bool = False,
        return_names: bool = False,
        transform: transforms = None,
    ) -> None:
        """
        Class constructor

        Parameters:
            psf_dirs (list):
                List of paths to `.npy` files containing information about the PSF.
            img_dirs (list):
                List of paths to image folders.
            n_images_per_psf (int):
                Number of images randomy choosed for each PSF.
                If None, all images will be concatenated to all PSFs.
                Default is None.
            seed (int):
                Random seed to choose images.
                Default is 42.
            gray (bool):
                Is a flag.
                If True, images will be converted to grayscale.
                Default is False.
            imagenet (bool):
                Is a flag.
                If True, cuts off the borders of the image, followed by bilinear interpolation to preserve the size.
                Default is False.
            return_names (bool):
                Is a flag.
                If True, return names of PSF and Image in each pair.
                Defalut is False.
            transform (torchvison.transforms):
                Image transforms from `torchvison` library.
                If None, uses convertation to `torch.tensor`.
                Default is None.
        """

        # apply transforms
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

        # bool parameters
        self.imagenet = imagenet
        self.return_names = return_names
        self.gray = gray

        # load psfs
        psf_cats = [np.load(psf_dir, allow_pickle=True) for psf_dir in psf_dirs]
        self.psfs = np.concatenate(
            [[psf["psf"] for psf in psf_cat] for psf_cat in psf_cats]
        )
        self.scas = np.concatenate(
            [
                [(psf["S"], psf["C"], psf["A"]) for psf in psf_cat]
                for psf_cat in psf_cats
            ]
        )

        # load and sort image paths
        self.img_paths = []
        for img_dir in img_dirs:
            for image_path in sorted(os.listdir(img_dir)):
                if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.abspath(os.path.join(img_dir, image_path))
                    self.img_paths.append(full_path)
        self.img_paths = np.array(self.img_paths)
        n_images = len(self.img_paths)

        # create pairs
        if n_images_per_psf is not None:
            # check number of images
            if n_images < n_images_per_psf:
                raise ValueError(
                    f"Number of images per PSF must be less then all images. Got n_images_per_psf={n_images_per_psf} and n_images={n_images}."
                )

            # setting seed
            np.random.seed(seed)
            self.pairs = []
            for psf_index in range(len(self.psfs)):
                # choosing random indexes from images
                random_indices = np.random.choice(
                    n_images, size=n_images_per_psf, replace=False
                )
                push_backed = [(psf_index, x) for x in random_indices]
                self.pairs.extend(push_backed)
        else:
            self.pairs = [
                (psf_index, img_index)
                for img_index in range(len(self.img_paths))
                for psf_index in range(len(self.psfs))
            ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        psf_index, image_index = self.pairs[idx]
        im_path = self.img_paths[image_index]

        psf = self.psfs[psf_index]
        psf = psf / np.sum(psf.flatten())
        psf = np.fft.fftshift(psf)
        psf = torch.FloatTensor(psf)
        psf = torch.unsqueeze(psf, dim=0)

        if self.gray:
            image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(im_path, cv2.IMREAD_COLOR_RGB)

        if image is None:
            raise ValueError(f"Image at {im_path} could not be loaded.")
        image = self.transform(image)

        if self.imagenet:
            crop = 3
            _, h, w = image.shape
            cropped = image[:, crop : h - crop, crop : w - crop].unsqueeze(0)
            cropped = F.interpolate(
                cropped, size=(h, w), mode="bilinear", align_corners=False
            )
            image = cropped.squeeze(0)

        if self.return_names:
            return psf, image, psf_index, im_path
        return psf, image
