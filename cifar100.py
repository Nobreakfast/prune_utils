import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import prune
from torch.utils.tensorboard import SummaryWriter
import argparse
import tqdm
from prune_utils.pai import *
from prune_utils.repair import repair_model_vgg
from prune_utils.initial import initialization
from data.core.dataloader import DataLoaderX

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def __getattr(model, name):
    name_list = name.split(".")
    ret = model
    for n in name_list:
        if n.isdigit():
            ret = ret[int(n)]
        else:
            ret = getattr(ret, n)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR100 Training")
    parser.add_argument("-s", "--save", help="save path", default="0")
    parser.add_argument("-p", "--prune", help="prune rate", type=float, default=0.0)
    parser.add_argument("-e", "--epoch", help="epoch", type=int, default=135)
    parser.add_argument("-a", "--algorithm", help="prune algorithm", default="nonprune")
    parser.add_argument("-r", "--restore", help="restore type", type=int, default=0)
    parser.add_argument(
        "-i",
        "--im",
        help="initialization method",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model",
        default=None,
    )
    args = parser.parse_args()

    # Set up TensorBoard writer with log directory
    save_path = f"logs/cifar100/{args.model}/{args.im}/{args.algorithm}/{args.prune:.2f}/r{args.restore}/no.{args.save}"
    writer = SummaryWriter(log_dir=save_path)

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

    trainset = torchvision.datasets.CIFAR100(
        root="~/Data/cifar100", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoaderX(trainset, batch_size=256, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root="~/Data/cifar100", train=False, download=True, transform=transform_test
    )
    testloader = DataLoaderX(testset, batch_size=256, shuffle=False, num_workers=4)

    if args.model == "vgg16":
        from models.vgg import vgg16

        model = vgg16()
    elif args.model == "vgg16_bn":
        from models.vgg import vgg16_bn

        model = vgg16_bn()
    else:
        raise ValueError("model not found")

    if args.im == "lsuv":
        from lsuv import lsuv_with_dataloader

        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = lsuv_with_dataloader(model, trainloader, device=device)
    else:
        initialization(model, args.im)

    if args.prune != 0.0:
        print(f"Original Sparsity: {cal_sparsity(model)}%")
        if args.algorithm == "rand":
            score_dict = rand(model)
            threshold = cal_threshold(score_dict, args.prune)
            apply_prune(model, score_dict, threshold)
        elif args.algorithm == "randn":
            score_dict = randn(model)
            threshold = cal_threshold(score_dict, args.prune)
            apply_prune(model, score_dict, threshold)
        elif args.algorithm == "snip":
            device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            score_dict = snip(model, trainloader)
            threshold = cal_threshold(score_dict, args.prune)
            apply_prune(model, score_dict, threshold)
            model = model.to(torch.device("cpu"))
        elif args.algorithm == "synflow":
            device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            example_data = torch.randn(1, 3, 32, 32)
            sign_dict = linearize(model)
            iterations = 100
            for i in range(iterations):
                prune_ratio = args.prune / iterations * (i + 1)
                score_dict = synflow(model, example_data)
                threshold = cal_threshold(score_dict, prune_ratio)
                if i != iterations - 1:
                    apply_prune(model, score_dict, threshold)
                    remove_mask(model)
                else:
                    nonlinearize(model, sign_dict)
                    apply_prune(model, score_dict, threshold)
            model = model.to(torch.device("cpu"))
        elif args.algorithm == "synflow_repair":
            device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            example_data = torch.randn(1, 3, 32, 32)
            sign_dict = linearize(model)
            iterations = 100
            for i in range(iterations):
                for module in model.modules():
                    if isinstance(module, nn.Conv2d):
                        sn = torch.linalg.norm(
                            module.weight.view(module.weight.shape[0], -1), ord=2
                        ).item()
                        # print(sn)
                        module.weight.data /= sn
                    elif isinstance(module, nn.Linear):
                        sn = torch.linalg.norm(module.weight, ord=2).item()
                        module.weight.data /= sn
                        # print(sn)
                prune_ratio = args.prune / iterations * (i + 1)
                score_dict = synflow(model, example_data)
                threshold = cal_threshold(score_dict, prune_ratio)
                if i != iterations - 1:
                    apply_prune(model, score_dict, threshold)
                    remove_mask(model)
                else:
                    nonlinearize(model, sign_dict)
                    apply_prune(model, score_dict, threshold)
            model = model.to(torch.device("cpu"))
        else:
            for name, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    prune.random_unstructured(m, name="weight", amount=args.prune)
                elif isinstance(m, nn.Linear):
                    prune.random_unstructured(m, name="weight", amount=args.prune)
        print(f"Pruned Sparsity: {cal_sparsity(model)}")
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                print(f"{name} sparsity: {cal_sparsity(m)}")

    if args.restore != 0:
        print("restoring !!!!")
        repair_model_vgg(model, args.restore)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 120, 140], gamma=0.1
    )

    # Training loop
    best = 0
    for epoch in tqdm.trange(args.epoch):
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
            train_loss = 0.0
            for data in trainloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                train_loss += criterion(outputs, labels).item()
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            test_loss = 0.0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_loss += criterion(outputs, labels).item()
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        test_accuracy = 100 * correct_test / total_test
        if test_accuracy > best:
            best = test_accuracy
            # torch.save(model.state_dict(), save_path + "/best.pth")

        generalization_gap = train_accuracy - test_accuracy
        generalization_loss = train_loss - test_loss
        writer.add_scalar("train accuracy", train_accuracy, epoch)
        writer.add_scalar("test accuracy", test_accuracy, epoch)
        writer.add_scalar("train loss", train_loss / len(trainloader), epoch)
        writer.add_scalar("test loss", test_loss / len(testloader), epoch)
        writer.add_scalar("generalization gap", generalization_gap, epoch)
        writer.add_scalar("generalization loss", generalization_loss, epoch)

    # Close TensorBoard writer
    writer.close()
    print("Finished Training")
    print("Best test accuracy: {:.2f}%".format(best))
