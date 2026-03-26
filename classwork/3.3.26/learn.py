import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class SquareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,1)


    def forward(self,x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x



        return x ** 2
    
model = SquareModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
X_train = torch.randn(1000, 1) * 3
print(X_train)

y_train = X_train ** 2

for epoch in range(10000):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    if torch.isnan(loss):
        print("nan")
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1 ) % 10 == 0:
        print(f"{epoch}, {loss.item()}")

    
    model.eval()

with torch.no_grad():
    x_test = torch.tensor([[2.0], [3.0], [-4.0]])
    result = model(x_test)
    for inp, out in zip(x_test, result):
        print(f"{inp.item():.3f}, {out.item():.3f}")