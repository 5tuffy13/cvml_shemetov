# super().__init__
import torch
from torch.utils.data import Dataset, DataLoader
class ImageDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = torch.randn(3,64,64)
        label = self.labels[idx]
        return image, label
    
dataset = ImageDataset(["a.png", "b.png", "c.png"], [0,1,2])
image, label = dataset[0]
print(len(dataset))

loader = DataLoader(dataset, batch_size = 3, shuffle = True)

for batch_idx, (data, target) in enumerate(loader):
    print(batch_idx, data.shape, target)

