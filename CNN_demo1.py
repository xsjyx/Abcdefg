import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split

# TODO: 讀取數據集，'train-images.pt' 和 'train-labels.csv'
images_raw = torch.load("/kaggle/input/moai-2025-training/train_images.pt", weights_only=False)
labels_raw = pd.read_csv('/kaggle/input/moai-2025-training/train_labels.csv')

# TODO: 歸一化數據集並轉換為 torch.Tensor
images_raw = images_raw.float()
images = (images_raw - images_raw.mean())/images_raw.std()

labels = torch.tensor(labels_raw['label'].values)

# TODO: 創建數據集，並按照 8:2 劃分成訓練集和驗證集

dataset =  TensorDataset(images,labels)
n = int(len(dataset) * 0.8)
train_dataset, val_dataset = random_split(dataset, [n, len(dataset)-n])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)



import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(loader, title):
    images, labels = next(iter(loader))
    images = images[:5]
    labels = labels[:5]
    
    plt.figure(figsize=(15, 3))
    plt.suptitle(title, fontsize=16)
    
    for i in range(5):
        plt.subplot(1, 5, i+1)
        img = images[i].squeeze()
        plt.imshow(img.numpy(), cmap='gray' if img.dim() == 2 else None)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 可視化訓練集樣本
visualize_samples(train_loader, "Training set")

# 可視化驗證集樣本
visualize_samples(val_loader, "Validation set")



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # TODO: 定義網絡層
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(32*7*7, 128)
        self.l2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # TODO: 定義前向傳播
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # x = torch.flatten(x,1)  => torch.flatten(input, start_dim=0, end_dim=-1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
print(model)



import torch.optim as optim
from sklearn.metrics import accuracy_score

# TODO: 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(5):
    # TODO: 訓練循環
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        train_loss = criterion(output, labels)
        train_loss.backward()
        optimizer.step()

        predict = output.argmax(dim=1)
        train_loss = train_loss.item()
        
        train_acc = accuracy_score(labels.numpy(), predict.numpy())
        
        # 以下是記錄損失函數和準確率的代碼，不用修改
        if batch_idx % 50 == 0:
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")
    
    # TODO: 驗證循環
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            val_loss = criterion(output, labels)
            
            predict = output.argmax(dim=1)
            val_loss = val_loss.item()

            val_acc = accuracy_score(labels.numpy(), predict.numpy())
           
    # 以下是記錄損失函數和準確率的代碼，不用修改 
    val_accs.append(val_acc)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss}, Val Accuracy: {val_acc}")





#Submission:
import pandas as pd
import torch

test_images = torch.load('/kaggle/input/moai-2025-training/test_images.pt', weights_only=True)
# TODO: 按照前面的方法歸一化
test_images = test_images.float()
test_images = ( test_images - test_images.mean() )/ test_images.std()

model.eval()
with torch.no_grad():
    test_images = test_images.to(device)
    outputs = model(test_images)
    predictions = outputs.argmax(dim=1)

df_test = pd.DataFrame({"label": predictions.cpu().numpy()})
df_test.to_csv("submission.csv", index_label="id")
