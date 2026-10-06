"""
Train Model
"""

import matplotlib.pyplot as plt
import os
import os.path as osp
import argparse
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.filters import grad
from utils.conv import fft_conv
from utils.imagetools import plot_images, add_psf_on_image, plot_loss
from utils.loss import PreCompensationLoss
from utils.dataset import PreCompensationDataset

from models.hqs_pc import HQS_PC


def plot_betas(betas: list, save_path: str) -> None:
    """Plot list of beta's"""
    plt.plot(betas, ".-", label=r"$\beta$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def train(model: HQS_PC, 
          train_dataset: PreCompensationDataset, 
          val_dataset: PreCompensationDataset,
          criterion: PreCompensationLoss,
          device: str = "cuda",
          lr: float = 1e-2,
          n_epochs: int = 10,
          batch_size: int = 2,
          shuffle: bool = True,
          verbose: bool = True,
          save_dir: str = "hqs/results"):
    """Train HQS-PC parameters"""
    model.to(device)

    checkpoints_path = osp.join(save_dir, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    print(f"[info] Save checkpoints in {checkpoints_path}")

    examples_path = osp.join(save_dir, "examples")
    os.makedirs(examples_path, exist_ok=True)
    print(f"[info] Save examples in {examples_path}")

    logs_path = osp.join(save_dir, "logs")
    os.makedirs(logs_path, exist_ok=True)
    print(f"[info] Save logs in {logs_path}")

    metrics = {"epoch": [], "train_loss": [], "val_loss": []}

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=shuffle)
    val_dataloader   = DataLoader(val_dataset, batch_size, shuffle=shuffle)

    total_train = len(train_dataloader)
    total_val   = len(val_dataloader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    for epoch in tqdm(range(n_epochs)):
        train_loss_epoch = 0
        model.train()
        for (psf, img) in tqdm(train_dataloader):
            img = img.to(device)
            psf = psf.to(device)

            # calculate pc-image
            pc = model(img, psf)

            # calculate loss
            loss = criterion(pc, img, psf)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clipping weights
            model.clip_weights()

            train_loss_epoch += loss.item()

        val_loss_epoch = 0

        with torch.inference_mode():
            model.eval()
            for (psf, img) in val_dataloader:
                img = img.to(device)
                psf = psf.to(device)

                # calculate pc-image
                # pc, reg = model(img, psf)
                pc = model(img, psf)

                # calculate loss
                loss = criterion(pc, img, psf)

                val_loss_epoch += loss.item()

            if verbose:
                for b in range(min(2, batch_size)):
                    save_path = osp.join(examples_path, f"example_batch_{b}.png")
                    image_and_psf = add_psf_on_image(img, psf, (80, 80))[b]
                    retinal_source = fft_conv(img, psf)[b]
                    retinal_pc = fft_conv(pc, psf)[b]
                    pc_show = pc[b]                    
                    plot_images([image_and_psf, retinal_source, retinal_pc, pc_show],
                                ["Original & PSF", "Retinal from Original", "Retinal from PC", "PC"],
                                figsize=(15, 15),
                                save_path=save_path)

        train_loss_epoch = train_loss_epoch / total_train
        val_loss_epoch = val_loss_epoch / total_val       

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss_epoch)
        metrics["val_loss"].append(val_loss_epoch)

        metrics_logs_path = osp.join(logs_path, "metrics_hqs.json")

        with open(metrics_logs_path, 'w') as f:
            json.dump({"metrics": metrics}, f, indent=4)

        plot_loss(metrics_logs_path, osp.join(logs_path, "loss_hqs.png"), False)
        plot_betas(model.beta.tolist(), osp.join(logs_path, "betas.png"))    
        
        checkpoint_path = os.path.join(checkpoints_path, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        if verbose:
            print(f"Loss on {epoch} epoch: train loss = {train_loss_epoch}, val loss = {val_loss_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="data/configs/train_config.json")
    parser.add_argument('--save_path', default='/home/devel/hqs/results')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda")
    opt = parser.parse_args()

    # read parameters from config
    with open(opt.config_path, 'r', encoding='utf-8') as f:
        prms = json.load(f)

    # make precompensation datasets
    train_dataset = PreCompensationDataset(**prms["data"]["train"])
    val_dataset   = PreCompensationDataset(**prms["data"]["val"])

    train_size = len(train_dataset)
    val_size   = len(val_dataset)

    print(f"[info] Succesfuly load data, got {train_size} images for train and {val_size} images for validation")

    # criterion
    filters = grad((prms["train"]["batch_size"], 1, 512, 512)).to(opt.device)
    criterion = PreCompensationLoss(**prms["loss"],
                                    filters=filters,
                                    reduction="mean")

    # make model
    model = HQS_PC(**prms["model"],
                   **prms["loss"],
                   filters=filters)
    # model = HQS_PC_wo_reg(n_iter=100)
    # state_dict = torch.load("hqs/results/weights/hqs_pc_15iter.pth", weights_only=True)
    # model.load_state_dict(state_dict)
    # model = torch.nn.DataParallel(model)
    # model = DWPC(tone_mapping="deep")

    # train model
    train(model, 
          train_dataset, 
          val_dataset,
          criterion,
          device=opt.device,
          **prms["train"],
          shuffle=False,
          verbose=opt.verbose,
          save_dir=opt.save_path)