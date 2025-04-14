# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
hidden_dim = 50
epochs = 3

# Create Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network
model = NN(input_size, num_classes, hidden_dim).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get correct shape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update weight
        optimizer.step()


# Check Accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    if loader.dataset.train:
        print("Accuracy on training data")
    else :
        print("Accuracy on test data")

    with torch.no_grad():
        for data, target in loader:

            data = data.reshape(data.shape[0], -1)
            scores = model(data)

            _, predicted = torch.max(scores, 1)
            num_correct += (predicted == target).sum()
            num_samples += predicted.size(0)

    print(f'Got {num_correct} / {num_samples} with accuracy {100 * num_correct / num_samples:.2f}%')

    model.train()
    return 100 * num_correct / num_samples

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)