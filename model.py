import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import os


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)  # 26x26x4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)  # 24x24x8
        self.pool = nn.MaxPool2d(2, 2)  # 12x12x8
        self.fc1 = nn.Linear(8 * 12 * 12, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 8 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and transform data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # Initialize model, loss function, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Batch: {batch_idx}, Loss: {loss.item():.4f}, "
                f"Accuracy: {100.*correct/total:.2f}%"
            )

    final_accuracy = 100.0 * correct / total

    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"mnist_model_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)

    return model, final_accuracy


if __name__ == "__main__":
    model, accuracy = train_model()
    print(f"Final Training Accuracy: {accuracy:.2f}%")
    print(f"Total Parameters: {count_parameters(model)}")
