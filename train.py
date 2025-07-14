# train_runner.py
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import wandb
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import dataclass
from tqdm.autonotebook import trange, tqdm

from dataloader import UWNetDataSet
from metrics_calculation import *
from models.UWnet_full import *
from models.block import *
from config import Config

__all__ = ["Trainer", "setup", "training"]

## TODO: Update config parameter names
## TODO: remove wandb steps
## TODO: Add comments to functions

@dataclass
class Trainer:
    models.UWnet_full: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module

    @torch.enable_grad()
    def train(self, train_loader, config, test_loader=None):
        self.models.UWnet_full.to(config.device)

        for epoch in trange(config.epochs, desc="Training Epochs"):
            epoch_loss = 0.0
            self.models.UWnet_full.train()

            for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
                x, y = batch
                x, y = x.to(config.device), y.to(config.device)

                self.optimizer.zero_grad()
                output = self.models.UWnet_full(x)
                loss = self.loss_fn(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"[Epoch {epoch+1}] Avg Loss: {epoch_loss / len(train_loader):.4f}")

            if test_loader and (epoch + 1) % config.eval_interval == 0:
                self.evaluate(config, test_loader)

            # Save snapshot
            if (epoch + 1) % config.snapshot_freq == 0:
                torch.save(self.models.UWnet_full.state_dict(), os.path.join(config.snapshot_dir, f"model_epoch_{epoch+1}.pth"))

    @torch.no_grad()
    def evaluate(self, config, test_loader):
        print("[Evaluation] Running evaluation...")
        self.models.UWnet_full.eval()

        for x, _, names in test_loader:
            x = x.to(config.device)
            out = self.models.UWnet_full(x)
            for i in range(x.size(0)):
                save_path = os.path.join(config.output_dir, names[i])
                torchvision.utils.save_image(out[i], save_path)

        uiqm = calculate_UIQM(config.output_dir)
        ssim, psnr = calculate_metrics_ssim_psnr(config.output_dir, config.gt_test_dir)
        print(f"[Evaluation Metrics] UIQM: {uiqm.mean():.4f}, SSIM: {ssim.mean():.4f}, PSNR: {psnr.mean():.4f}")

def setup(config):
    models.UWnet_full = UWnet().to(config.device)
    optimizer = torch.optim.Adam(models.UWnet_full.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.ToTensor()
    ])

    train_set = DummyDataset(transform=transform, is_train=True)
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True)

    if config.test_mode:
        test_set = DummyDataset(transform=transform, is_train=False)
        test_loader = DataLoader(test_set, batch_size=config.test_batch_size, shuffle=False)
    else:
        test_loader = None

    trainer = Trainer(models.UWnet_full, optimizer, loss_fn)
    return trainer, train_loader, test_loader

def training():
    config = Config()
    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    trainer, train_loader, test_loader = setup(config)
    trainer.train(train_loader, config, test_loader)

if __name__ == "__main__":
    training()

