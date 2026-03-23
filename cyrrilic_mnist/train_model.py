from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

cur_path = Path(__file__).parent

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")





class CyrillicDataset(Dataset):
    def __init__(self,path):
        #data loading
        path = Path(path)
        num_classes = -1
        self.dataset = []
        self.alphabet = ""
        for cls in sorted(path.glob("*")):
            if str(cls.name)[0] != ".": # to skip .DS_Store
                num_classes+=1
                self.alphabet+=str(cls)[-1]
                for letter in cls.glob("*.png"):
                    self.dataset.append((Image.open(letter),num_classes))

        
    
        

    def __getitem__(self, index):
        image = self.dataset[index][0].copy()
        augments = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 1), 10)])
        return augments(image), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)

dataset = CyrillicDataset(f"{cur_path}/Cyrillic")
train, test = train_test_split(dataset, test_size=0.1, shuffle=True)



train_loader = DataLoader(dataset=train , shuffle=True, batch_size=32)
test_loader = DataLoader(dataset=test , shuffle=True, batch_size=1024)
#278*278



class CyrillicCNN(torch.nn.Module):
    def __init__(self):
        super(CyrillicCNN, self).__init__()
        self.relu = torch.nn.ReLU()

        self.cn1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.maxpool1 = torch.nn.MaxPool2d(2,2) # 128 * 128

        self.cn2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.maxpool2 = torch.nn.MaxPool2d(2,2,padding=1) # 64 * 64

        self.cn3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.maxpool3 = torch.nn.MaxPool2d(2,2) # 32 * 32

        self.cn4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.maxpool4 = torch.nn.MaxPool2d(4,4) # 16 * 16

        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.45)
        self.fc1 = torch.nn.Linear(2*2*256, 34)


    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.cn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.cn3(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.relu(x)

        x = self.cn4(x)
        x = self.bn4(x)
        x = self.maxpool4(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    


model = CyrillicCNN().to(device)

total_params = sum(p.numel() for p in model.parameters())

print(f"{total_params=}")


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 10**-4)
 
num_epochs = 15

train_loss = []
train_acc = []

model_path = cur_path / "model.pth"

if not model_path.exists():
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (sample,label) in enumerate(train_loader):
            sample, label = (sample.to(device), label.to(device))
            optimizer.zero_grad()
            outputs = model(sample)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        epoch_loss = run_loss/len(train_loader)
        epoch_acc = 100 * (correct/total)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Epoch {epoch}, {epoch_loss:=.3f}, {epoch_acc:=.3f}")

    torch.save(model.state_dict(), model_path)
    plt.figure()
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_loss)
    plt.subplot(122)
    plt.title("Acc")
    plt.plot(train_acc)
    plt.savefig(cur_path / "train.png")
    plt.show()
            

else:
    model.load_state_dict(torch.load(model_path))
