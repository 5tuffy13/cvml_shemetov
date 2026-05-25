from unet_road import RoadDataset, UNet
import torch
from pathlib import Path
import matplotlib.pyplot as plt


root = Path(__file__).parent

model = UNet()

model.load_state_dict(torch.load(root / "model.pth"))

model.eval()

ds = RoadDataset(root / "roads")

with torch.no_grad():
    img, mask = ds[2]
    img = img.unsqueeze(0)
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mask.permute(1,2,0).squeeze(0))
    plt.subplot(132)
    plt.imshow(torch.sigmoid(model(img)).squeeze(0).permute(1,2,0))
    plt.subplot(133)
    plt.imshow(mask.permute(1,2,0).squeeze(0) - torch.sigmoid(model(img)).squeeze(0).permute(1,2,0))
    plt.show()
   