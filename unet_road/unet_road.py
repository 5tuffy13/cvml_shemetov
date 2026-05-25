import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt




root = Path(__file__).parent
path = root / "roads"

class RoadDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.images_paths = path / "images"
        self.masks_paths = path / "masks"
        self.images = sorted(list(self.images_paths.glob('*.png')))
        self.masks = sorted(list(self.masks_paths.glob('*.png')))
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        # почему не работает
        # image = np.array(image, dtype='f4')
        image = np.array(image) / 255.0
        
        mask = Image.open(self.masks[idx]).convert('L')
        mask = np.array(mask, dtype='f4')
        mask = (mask == 82).astype('f4')
        mask = np.expand_dims(mask, axis=0) # 1, h, w
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() # c, h, w
        mask = torch.from_numpy(mask).float() # 1, h, w
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3 , 1 , 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3 , 1 , 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,features=[64, 128, 256, 512]):
        super().__init__()
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        for n in features:
            self.downscale.append(DoubleConv(in_channels, n))
            in_channels = n

        for n in reversed(features):
            self.upscale.append(
                nn.ConvTranspose2d(n*2, n, 2,2)
            )
            self.upscale.append(DoubleConv(n*2, n))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.result = nn.Conv2d(features[0], out_channels,1)


    def forward(self,x):
        skips = []
        
        for ds in self.downscale:
            x = ds(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for idx in range(0,len(self.upscale),2):
            x = self.upscale[idx](x)
            skip = skips[idx//2]
            cx = torch.cat((skip,x),dim=1)
            x = self.upscale[idx+1](cx)
        return self.result(x)
    

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        p_area = pred_sig.view(-1)
        t_area = target.view(-1)
        intersection = (p_area * t_area).sum()
        return (1 - (2 * intersection + 1) / (p_area.sum() + t_area.sum() + 1))

        

if __name__ == "__main__":


    device = torch.device("mps")
    model = UNet().to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))

    ds = RoadDataset(path)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)



    model.train()

    epochs = 15

    torch.mps.empty_cache()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for road, mask in dl:
            road = road.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output = model(road)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dl)
        print(f"{epoch=},   {avg_loss=:.2f}")

    torch.save(model.state_dict(), root / "model.pth")

