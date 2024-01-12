import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import prune
from torch.utils.tensorboard import SummaryWriter
import argparse
import tqdm
import models.resnet as resnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("-i", "--im", help="initialization method", default="")
    parser.add_argument("-s", "--save", help="save path", default="./logs/test")
    parser.add_argument("-l", "--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("-p", "--prune", help="prune rate", type=float, default=0.0)
    parser.add_argument("-r", "--restore", help="restore rate", type=float, default=0.0)

    args = parser.parse_args()

    # Set up TensorBoard writer with log directory
    writer = SummaryWriter(log_dir=args.save)

    # Load CIFAR-10 dataset
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="~/Data/cifar10", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="~/Data/cifar10", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    # Initialize the model, loss function, and optimizer
    model = resnet.resnet20()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if args.im == "xavier":
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif args.im == "kaiming_out":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif args.im == "kaiming_in":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif args.im == "haocheng":
                m.weight.data = torch.randn(m.weight.shape)
                m.weight.data /= torch.linalg.norm(
                    m.weight.view(m.weight.shape[0], -1), ord=2
                )
            else:
                print("No initialization method specified")
        elif isinstance(m, nn.Linear):
            m.weight.data = torch.randn(m.weight.shape)
            m.weight.data /= torch.linalg.norm(m.weight, ord=2)
            # if args.im == "xavier":
            #     nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            # elif args.im == "kaiming_out":
            #     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # elif args.im == "kaiming_in":
            #     nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            # elif args.im == "haocheng":
            #     m.weight.data = torch.randn(m.weight.shape)
            #     m.weight.data /= torch.linalg.norm(m.weight, ord=2)
            # else:
            #     print("No initialization method specified")
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    if args.prune != 0.0:
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                prune.l1_unstructured(m, name="weight", amount=args.prune)
                if m.bias is not None:
                    prune.l1_unstructured(m, name="bias", amount=args.prune)

    if args.restore != 0.0:
        sn_0 = torch.linalg.norm(
            model.conv1.weight.data.view(model.conv1.weight.shape[0], -1), ord=2
        )
        sn_11 = torch.linalg.norm(
            model.layer1[0].conv1.weight.data.view(
                model.layer1[0].conv1.weight.shape[0], -1
            ),
            ord=2,
        )
        sn_12 = torch.linalg.norm(
            model.layer1[0].conv2.weight.data.view(
                model.layer1[0].conv2.weight.shape[0], -1
            ),
            ord=2,
        )
        sn_21 = torch.linalg.norm(
            model.layer2[0].conv1.weight.data.view(
                model.layer2[0].conv1.weight.shape[0], -1
            ),
            ord=2,
        )
        sn_22 = torch.linalg.norm(
            model.layer2[0].conv2.weight.data.view(
                model.layer2[0].conv2.weight.shape[0], -1
            ),
            ord=2,
        )
        sn_31 = torch.linalg.norm(
            model.layer3[0].conv1.weight.data.view(
                model.layer3[0].conv1.weight.shape[0], -1
            ),
            ord=2,
        )
        sn_32 = torch.linalg.norm(
            model.layer3[0].conv2.weight.data.view(
                model.layer3[0].conv2.weight.shape[0], -1
            ),
            ord=2,
        )
        sn_4 = torch.linalg.norm(model.fc.weight.data, ord=2)
        bn_0 = (0.5**0.5) / sn_0
        bn_12 = (bn_0) / (sn_11 * sn_12)
        bn_22 = (bn_0 + bn_12) / (sn_21 * sn_22)
        bn_32 = (bn_0 + bn_12 + bn_22) / (sn_31 * sn_32)
        model.bn1.weight.data *= bn_0
        model.layer1[0].bn2.weight.data *= bn_12
        model.layer2[0].bn2.weight.data *= bn_22
        model.layer3[0].bn2.weight.data *= bn_32
        model.fc.weight.data *= args.restore / sn_4
        print("sn_0: ", sn_0)
        print("sn_11: ", sn_11)
        print("sn_12: ", sn_12)
        print("sn_21: ", sn_21)
        print("sn_22: ", sn_22)
        print("sn_31: ", sn_31)
        print("sn_32: ", sn_32)
        print("sn_4: ", sn_4)
        print("bn_0: ", bn_0)
        print("bn_12: ", bn_12)
        print("bn_22: ", bn_22)
        print("bn_32: ", bn_32)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=1e-3, max_lr=0.1, step_size_up=5, step_size_down=15
    # )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 120, 140], gamma=0.1
    )

    # Training loop
    for epoch in range(160):
        running_loss = 0.0
        model.train()
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
        scheduler.step()

        # Calculate accuracy on train and test sets
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0
        model.eval()
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
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
