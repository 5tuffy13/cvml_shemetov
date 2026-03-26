# super().__init__
# pip install pooch, torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.datasets import face
from torchvision.transforms import v2 as transforms
import matplotlib.pyplot as plt
class ImageDataset(Dataset):
    def __init__(self, n_images = 100):
        self.n = n_images
        self.image = face()
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        image = self.image.copy()
        augments = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToImage(),
            transforms.RandomAffine(5, (0.1, 0.1), (0.5, 1), 10)


        ])
        return augments(image), idx
    
dataset = ImageDataset()
image, label = dataset[0]
print(len(dataset))

loader = DataLoader(dataset, batch_size = 5, shuffle = True)


plt.figure(figsize=(7,7))
plt.ion()
for batch_idx, (data, target) in enumerate(loader):
    print(batch_idx, data.shape, target)

    for i in range(data.shape[0]):
        plt.clf()
        plt.imshow(data[i].numpy().transpose((1,2,0)))
        plt.pause(0.2)