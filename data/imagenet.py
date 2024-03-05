import os

import torch
import torchvision
import torchvision.transforms as transforms


def imagenet(batch_size, path, workers):
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(225),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(path, "train"), transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    testset = torchvision.datasets.ImageFolder(
        root=os.path.join(path, "val"), transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size * 8, shuffle=False, num_workers=workers
    )

    return [trainloader, testloader]
