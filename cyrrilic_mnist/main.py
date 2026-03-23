from train_model import CyrillicCNN, test, dataset, test_loader
import torch
from pathlib import Path
from torch.utils.data import DataLoader


cur_path = Path(__file__).parent

model_path = cur_path / "model.pth"

test_loader = DataLoader(dataset=test , shuffle=True, batch_size = 1024)


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = CyrillicCNN().to(device)
model.load_state_dict(torch.load(model_path))



model.eval()
correct = 0
whole = 0
for i in range(1024):
    it = iter(test_loader)
    images, labels = next(it)
    image = images[i].unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        if dataset.alphabet[labels[i]] == dataset.alphabet[predicted.cpu().item()]:
            correct += 1
        whole +=1
        if i > 1000:
            print(f"True - {dataset.alphabet[labels[i]]}\nPred - {dataset.alphabet[predicted.cpu().item()]}")
print("Accuracy: ",correct/whole * 100)