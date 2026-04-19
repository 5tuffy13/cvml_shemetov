import torch 
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



foldr = Path(__file__).parent

device = "mps"


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])




def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    modelB0 = torchvision.models.efficientnet_b0(weights=weights)
    for p in modelB0.features.parameters():
        p.requires_grad = False

    in_features = modelB0.classifier[1].in_features
    modelB0.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features,3)

    )
    modelB0.load_state_dict(torch.load(foldr / "efnetB0.pth"))

    

    weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
    modelB1 = torchvision.models.efficientnet_b1(weights=weights)
    for p in modelB1.features.parameters():
        p.requires_grad = False

    in_features = modelB1.classifier[1].in_features
    modelB1.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features,3)

    )
    modelB1.load_state_dict(torch.load(foldr / "efnetB1.pth"))

    weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
    modelB2 = torchvision.models.efficientnet_b2(weights=weights)
    for p in modelB2.features.parameters():
        p.requires_grad = False

    in_features = modelB2.classifier[1].in_features
    modelB2.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features,3)

    )
    modelB2.load_state_dict(torch.load(foldr / "efnetB2.pth"))

    return modelB0.to(device), modelB1.to(device), modelB2.to(device)

def predict(model, frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to("mps") # [[[]]]
    with torch.no_grad():
        predicted = torch.argmax(model(tensor), dim = 1) 
    return predicted



modelB0, modelB1, modelB2 = build_model()

ds_path = foldr / "dataset/val"
class_names = sorted([f.name for f in ds_path.glob("*") if not f.name.startswith(".")])

test = []
ncls = -1
for cls in sorted(ds_path.glob("*")):
    if not cls.name.startswith("."):
        ncls += 1
        for img in sorted(cls.glob("*")):
            if not img.name.startswith("."):
                test.append((cv2.imread(str(img)), ncls))

y_true = []
y_pred_b0 = []
y_pred_b1 = []
y_pred_b2 = []

for img, label in test:
    y_true.append(label)
    
    y_pred_b0.append(predict(modelB0, img).item())
    y_pred_b1.append(predict(modelB1, img).item())
    y_pred_b2.append(predict(modelB2, img).item())


def show_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()
show_cm(y_true, y_pred_b0, "EfficientNet B0")
show_cm(y_true, y_pred_b1, "EfficientNet B1")
show_cm(y_true, y_pred_b2, "EfficientNet B2")

from sklearn.metrics import accuracy_score
print(f"Acc B0: {accuracy_score(y_true, y_pred_b0)}")
print(f"Acc B1: {accuracy_score(y_true, y_pred_b1)}")
print(f"Acc B2: {accuracy_score(y_true, y_pred_b2)}")