
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataset import UWImageDataset
from models.UWnet_full import UWnet
from losses.loss import L1Loss
from utils.utils import save_image_tensor
import os

# 设置参数
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "path_to_training_images"
SAVE_PATH = "output_images"

# 加载数据
dataset = UWImageDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型、损失、优化器
model = UWnet().to(DEVICE)
criterion = L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练循环
for epoch in range(EPOCHS):
    for i, inputs in enumerate(dataloader):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
            save_image_tensor(outputs, os.path.join(SAVE_PATH, f"epoch_{epoch}_step_{i}.png"))
