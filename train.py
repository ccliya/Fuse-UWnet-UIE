
# train_runner.py
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import wandb
import argparse

#from dataloader import UWNetDataSet
from torchvision import transforms
from dataclasses import dataclass
from tqdm.autonotebook import trange, tqdm
from torch.utils.data import DataLoader
from models.UWnet_full import UWnet
from models.deconv import DEConv

from dataloader import UWNetDataSet
from metrics_calculation import *
from models.UWnet_full import UWnet
from models.blocks import ConvBlock, ELA, MambaBlock, BAM, AdaIN, DenSoA
from config import config

__all__ = ["Trainer", "setup", "training"]

## TODO: Update config parameter names
## TODO: remove wandb steps
## TODO: Add comments to functions

@dataclass
#@dataclass
class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @torch.enable_grad()
    def train(self, train_loader, config, test_loader=None):
        self.model.to(config.device)

        for epoch in trange(config.epochs, desc="Training Epochs"):
            epoch_loss = 0.0
            self.model.train()

            for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
                x, y, _ = batch
                x, y = x.to(config.device), y.to(config.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss_fn(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"[Epoch {epoch+1}] Avg Loss: {epoch_loss / len(train_loader):.4f}")

            if test_loader and (epoch + 1) % config.eval_interval == 0:
                self.evaluate(config, test_loader)

            # Save snapshot
            if (epoch + 1) % config.snapshot_freq == 0:
                #torch.save(self.models.UWnet_full.state_dict(), os.path.join(config.snapshot_dir, f"model_epoch_{epoch+1}.pth"))
                torch.save(self.model.state_dict(), os.path.join(config.snapshot_dir, f"model_epoch_{epoch+1}.pth"))

    @torch.no_grad()
    def evaluate(self, config, test_loader):
        print("[Evaluation] Running evaluation...")

    # 设置模型为评估模式
        self.model.eval()  # ✅ 修改为 self.model（不要写 self.models.UWnet_full）

    # 创建输出目录（如果不存在）
        os.makedirs(config.output_images_path, exist_ok=True)

    # 遍历测试集
        for x, _, names in test_loader:
            x = x.to(config.device)
            out = self.model(x)

        # 保存每一张输出图像
            for i in range(x.size(0)):
                name_i = names[i]
                if not name_i.lower().endswith(".jpg"):
                    name_i += ".jpg"  # 默认加后缀，避免没有扩展名
                save_path = os.path.join(config.output_images_path, name_i)
                torchvision.utils.save_image(out[i], save_path)

    # 计算评估指标
        uiqm = calculate_UIQM(config.output_images_path)
        ssim, psnr = calculate_metrics_ssim_psnr(config.output_images_path, config.test_clear_image_path)

    # 输出评估指标
        print(f"[Evaluation Metrics] UIQM: {uiqm.mean():.4f}, SSIM: {ssim.mean():.4f}, PSNR: {psnr.mean():.4f}")



#from dataloader import UWNetDataSet  # 确保你已经导入了真实的数据集类

from dataloader import UWNetDataSet

def setup(config):
    model = UWnet().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize((config.resize, config.resize)),
        transforms.ToTensor()
    ])

    # 加载训练数据
    train_set = UWNetDataSet(
        raw_image_path=config.train_raw_image_path,
        clear_image_path=config.train_clear_image_path,
        transform=transform,
        is_train=True
    )
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True)

    # 加载测试数据
    test_loader = None
    if config.test_mode:
        test_set = UWNetDataSet(
            raw_image_path=config.test_raw_image_path,
            clear_image_path=config.test_clear_image_path,
            transform=transform,
            is_train=False
        )
        test_loader = DataLoader(test_set, batch_size=config.test_batch_size, shuffle=False)
    else:
        test_loader = None

    trainer = Trainer(model, optimizer, loss_fn)
    #return trainer, train_loader, test_loader
    return model, trainer, train_loader, test_loader

def training():
    #创建必要目录
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.output_images_path, exist_ok=True)
#初始化wandb等日志（可选）
    wandb.init(project="underwater_image_enhancement_UWNet")
    wandb.config.update({k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v)}, allow_val_change=True)
#准备数据和模型   
    model, trainer, train_loader, test_loader = setup(config)
#开始训练
    trainer.train(train_loader, config, test_loader)

    print("==================")
    print("Training complete!")
    print("==================")

if __name__ == "__main__":

   
    training()

