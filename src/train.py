
# Training a Handwritten Kanji Recognition Model with ETL9B (PyTorch)

# 1. Dataset Acquisition & Preprocessing (ETL9B)
import struct
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ETL9BDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        with open(file_path, 'rb') as f:
            while True:
                s = f.read(512)
                if not s:
                    break
                try:
                    r = struct.unpack('>2H4s504s', s)
                    i = Image.frombytes('F', (128, 127), r[3], 'bit', 4)
                    i = i.convert('L')
                    label = r[0]  # JIS code
                    self.data.append(i)
                    self.labels.append(label)
                except Exception as e:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 2. Transformations and DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ETL9BDataset('ETL9B/ETL9B_1', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. CNN Model for Classification
import torch
import torch.nn as nn
import torch.nn.functional as F

class KanjiCNN(nn.Module):
    def __init__(self, num_classes=3036):
        super(KanjiCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. Training Loop
import torch.optim as optim

model = KanjiCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# 5. Save Model
torch.save(model.state_dict(), 'kanji_cnn.pth')
