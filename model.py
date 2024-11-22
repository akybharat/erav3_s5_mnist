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
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3, padding=1)  # 28x28x16
        self.conv2 = nn.Conv2d(30, 40, kernel_size=3, padding=1)  # 28x28x32
        self.pool = nn.MaxPool2d(2, 2)  # 14x14x32
        self.conv3 = nn.Conv2d(40, 16, kernel_size=3, padding=1)  # 14x14x16
        self.dropout1 = nn.Dropout(0.03)
        self.dropout2 = nn.Dropout(0.03)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)
        self.batch_norm1 = nn.BatchNorm2d(30)
        self.batch_norm2 = nn.BatchNorm2d(40)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.batch_norm1(self.relu(self.conv1(x))))  # 14x14x16
        x = self.dropout1(x)
        x = self.batch_norm2(self.relu(self.conv2(x)))  # 14x14x32
        x = self.pool(self.batch_norm3(self.relu(self.conv3(x))))  # 7x7x16
        x = self.dropout2(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and transform data with augmentation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0, translate=(0.12, 0.12), scale=(0.85, 1.15)
            ),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=2
    )

    # Initialize model, loss function, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=1,
        steps_per_epoch=len(trainloader),
        div_factor=25,
        final_div_factor=1000,
    )

    # Training
    model.train()
    correct = 0
    total = 0

    # Single epoch training
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # Added scheduler step

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
