import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

train_data = MNIST(
    "./data/",
    train=True,
    download=True,
    transform=T.Compose([T.ToTensor()])
)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

valid_data = MNIST(
    "./data/",
    train=False,
    download=True,
    transform=T.Compose([T.ToTensor()])
)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = MyNet()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

train_loss_history = []
valid_loss_history = []
for epoch in range(10):
    loss_sum = 0
    print("Epoch:", epoch)
    net.train()
    for x, y in train_loader:
        x = x.reshape(len(x), 28 * 28)
        h = net(x)
        one_hot_y = F.one_hot(y, num_classes=10)
        loss = loss_fn(h, one_hot_y.to(dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().numpy()
    print(loss_sum / len(train_loader))
    train_loss_history.append(loss_sum / len(train_loader))

    net.eval()
    loss_sum = 0
    for x, y in valid_loader:
        x = x.reshape(len(x), 28 * 28)
        h = net(x)
        one_hot_y = F.one_hot(y, num_classes=10)
        loss = loss_fn(h, one_hot_y.to(dtype=torch.float32))
        loss_sum += loss.detach().numpy()
    print(loss_sum / len(valid_loader))
    valid_loss_history.append(loss_sum / len(valid_loader))
plt.plot(train_loss_history)
plt.plot(valid_loss_history)
plt.show()
