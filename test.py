import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("-i", "--im", help="initialization method", default="")
    parser.add_argument("-s", "--save", help="save path", default="./logs/test")
    parser.add_argument("-l", "--lr", help="learning rate", default=0.001)

    args = parser.parse_args()

    # Set up TensorBoard writer with log directory
    writer = SummaryWriter(log_dir=args.save)

    # Define the linear model
    class LinearModel(nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.fc1 = nn.Linear(3072, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="~/Data/cifar10", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="~/Data/cifar10", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    # Initialize the model, loss function, and optimizer
    model = LinearModel()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001
    )

    # Training loop
    for epoch in range(80):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                writer.add_scalar(
                    "training loss", running_loss / 200, epoch * len(trainloader) + i
                )
                running_loss = 0.0

        # Calculate accuracy on train and test sets
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        test_accuracy = 100 * correct_test / total_test

        writer.add_scalar("train accuracy", train_accuracy, epoch)
        writer.add_scalar("test accuracy", test_accuracy, epoch)

    # Close TensorBoard writer
    writer.close()
