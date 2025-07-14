
import torch
from torch.utils.data import DataLoader
from models.UWnet_full import UWnet
from dataset import DummyDataset
from config import Config

def train():
    model = UWnet().to(Config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    loss_fn = torch.nn.MSELoss()

    loader = DataLoader(DummyDataset(), batch_size=Config.batch_size)

    for epoch in range(Config.epochs):
        for x, y in loader:
            x, y = x.to(Config.device), y.to(Config.device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
