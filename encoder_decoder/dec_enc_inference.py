from classwork.encoder_decoder.train import (Decoder, Encoder, ImageDataset)
import torch
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).parent

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load(root / "encoder.pth"))
decoder.load_state_dict(torch.load(root / "decoder.pth"))

encoder.requires_grad_(False)
decoder.requires_grad_(False)

dataset = ImageDataset(10, 256, 4)
image, _ = dataset[16]

latent = encoder(image.unsqueeze(0))
result = decoder(latent)


plt.subplot(131)
plt.imshow(image.squeeze())
plt.subplot(132)
plt.imshow(result.squeeze())
plt.subplot(133)
plt.imshow(image.squeeze() - result.squeeze())
plt.show()