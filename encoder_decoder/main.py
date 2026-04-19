import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
class ImageDataset(Dataset):
    def __init__(self, n=200, size=128):
        self.n = n
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        super().__init__()

    def __len__(self):
        return self.n
    

    def __getitem__(self, index):
        image = Image.new('L', (self.size, self.size), color = 255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        text = "ABC"

        x = np.random.randint(10, self.size - 40)
        y = np.random.randint(10, self.size - 40)
        draw.text((x,y), text, fill=0, font=font)
        tensor = self.transform(image)
        return tensor, tensor
    

dataset = ImageDataset()

class Encoder(nn.Module):
    def __init__(self, latent = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, stride = 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,64, stride = 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128, stride = 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256, stride = 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x