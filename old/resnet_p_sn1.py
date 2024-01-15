import pandas as pd
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


# pruning the L1 high value
class PruningL1HighValue(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        self.amount = amount

    def compute_mask(self, t, default_mask):
        numtoprune = int(self.amount * t.nelement())
        if numtoprune == 0:
            return default_mask
        mask = default_mask.clone()
        iteration = 10
        for i in range(iteration):
            sn = torch.linalg.norm((t * mask).view(t.size(0), -1), ord=2)
            if sn >= 1:
                tmp_numtoprune = int(numtoprune / iteration)
                mask.view(-1)[
                    torch.topk((t * mask).abs().view(-1), tmp_numtoprune, largest=True)[
                        1
                    ]
                ] = 0
            else:
                tmp_numtoprune = int(numtoprune * (i + 1) / iteration)
                mask.view(-1)[
                    torch.topk(
                        (t * mask).abs().view(-1), tmp_numtoprune, largest=False
                    )[1]
                ] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        return super(PruningL1HighValue, cls).apply(module, name, amount)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("-i", "--im", help="initialization method", default="")
    parser.add_argument("-s", "--save", help="save path", default="./logs/test")
    parser.add_argument(
        "-l", "--load", help="load path", default="./logs/resnet/baseline/best.pth"
    )
    parser.add_argument("-r", "--lr", help="learning rate", type=float, default=0.1)
    parser.add_argument("-p", "--prune", help="prune rate", type=float, default=0.0)

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
    model.load_state_dict(torch.load(args.load))
    data = pd.DataFrame(
        columns=["name", "in_shape", "out_shape", "sn", "var", "std", "sv", "sv_n"]
    )
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_shape = (
                module.weight.data.shape[1]
                * module.weight.data.shape[2]
                * module.weight.data.shape[3]
            )
            out_shape = module.weight.data.shape[0]
            sn = torch.linalg.norm(
                module.weight.data.view(module.weight.data.size(0), -1), ord=2
            ).item()
            var = module.weight.data.var().item()
            std = module.weight.data.std().item()
            sv = sn / std
            sv_n = sn / std / ((in_shape / 2) ** 0.5)
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "name": name,
                            "in_shape": in_shape,
                            "out_shape": out_shape,
                            "sn": sn,
                            "var": var,
                            "std": std,
                            "sv": sv,
                            "sv_n": sv_n,
                        },
                        index=[0],
                    ),
                ]
            )
    path = args.save + "/before.csv"
    data.to_csv(path)
    print(data)

    if args.prune != 0.0:
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                PruningL1HighValue.apply(m, "weight", args.prune)
                print(torch.sum(m.weight_mask))

    data = pd.DataFrame(
        columns=["name", "in_shape", "out_shape", "sn", "var", "std", "sv", "sv_n"]
    )
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_shape = (
                module.weight.data.shape[1]
                * module.weight.data.shape[2]
                * module.weight.data.shape[3]
            )
            out_shape = module.weight.data.shape[0]
            sn = torch.linalg.norm(
                module.weight.data.view(module.weight.data.size(0), -1), ord=2
            ).item()
            var = module.weight.data.var().item()
            std = module.weight.data.std().item()
            sv = sn / std
            sv_n = sn / std / ((in_shape / 2) ** 0.5)
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "name": name,
                            "in_shape": in_shape,
                            "out_shape": out_shape,
                            "sn": sn,
                            "var": var,
                            "std": std,
                            "sv": sv,
                            "sv_n": sv_n,
                        },
                        index=[0],
                    ),
                ]
            )
    path = args.save + "/after.csv"
    data.to_csv(path)
    print(data)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 120, 140], gamma=0.1
    )

    # Training loop
    best = 0
    for epoch in tqdm.trange(160):
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
        if test_accuracy > best:
            best = test_accuracy
            torch.save(model.state_dict(), args.save + "/best_p.pth")

        writer.add_scalar("train accuracy", train_accuracy, epoch)
        writer.add_scalar("test accuracy", test_accuracy, epoch)

    model.load_state_dict(torch.load(args.save + "/best_p.pth"))
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m, "weight")
    data = pd.DataFrame(
        columns=["name", "in_shape", "out_shape", "sn", "var", "std", "sv", "sv_n"]
    )
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_shape = (
                module.weight.data.shape[1]
                * module.weight.data.shape[2]
                * module.weight.data.shape[3]
            )
            out_shape = module.weight.data.shape[0]
            sn = torch.linalg.norm(
                module.weight.data.view(module.weight.data.size(0), -1), ord=2
            ).item()
            var = module.weight.data.var().item()
            std = module.weight.data.std().item()
            sv = sn / std
            sv_n = sn / std / ((in_shape / 2) ** 0.5)
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "name": name,
                            "in_shape": in_shape,
                            "out_shape": out_shape,
                            "sn": sn,
                            "var": var,
                            "std": std,
                            "sv": sv,
                            "sv_n": sv_n,
                        },
                        index=[0],
                    ),
                ]
            )
    path = args.save + "/after_train.csv"
    data.to_csv(path)
    print(data)
    torch.save(model.state_dict(), args.save + "/best.pth")
    # Close TensorBoard writer
    writer.close()
    print("Finished Training")
    print("Best test accuracy: {:.2f}%".format(best))
