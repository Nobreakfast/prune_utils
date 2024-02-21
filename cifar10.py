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
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("-s", "--save", help="save path", default="test")
    parser.add_argument("-p", "--prune", help="prune rate", type=float, default=0.0)
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
    # only works in resnet_res
    parser.add_argument(
        "--alpha",
        help="alpha",
        type=float,
        default=1.0,
    )
    # only works in resnet_res
    parser.add_argument(
        "--beta",
        help="beta",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    # Set up TensorBoard writer with log directory
    # save_path = f"logs/cifar10/{args.model}/{args.im}_p{args.prune:.2f}_r{args.restore}_no.{args.save}"
    save_path = f"logs/cifar10/{args.model}/{args.im}/{args.algorithm}/{args.prune:.2f}r{args.restore}/no.{args.save}"
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

    trainset = torchvision.datasets.CIFAR10(
        root="~/Data/cifar10", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR10(
        root="~/Data/cifar10", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=4
    )

    if args.model == "resnet20_wobn":
        from models.resnet_wobn import resnet20

        model = resnet20()
    elif args.model == "resnet20":
        from models.resnet import resnet20

        model = resnet20()
    elif args.model == "resnet20_res":
        from models.resnet_res import resnet20

        model = resnet20(args.alpha, args.beta)
    elif args.model == "vgg16":
        from models.vgg16 import VGG16

        model = VGG16()
    elif args.model == "vgg16_bn":
        from models.vgg16 import VGG16_BN

        model = VGG16_BN()

    elif args.model[:2] == "fc":
        import models.fc as fc

        # extract the number of layers from the model name like fc3 fc20 fc50_wobn
        num_layers = int(args.model.split("_")[0][2:])

        if args.model[-4:] == "wobn":
            model = fc.FCN_WOBN(num_layers)
        else:
            model = fc.FCN(num_layers)

    elif args.model[:4] == "conv":
        import models.conv as conv

        # extract the number of layers from the model name like conv3 conv20 conv50_wobn
        num_layers = int(args.model.split("_")[0][4:])
        if args.model[-4:] == "wobn":
            model = conv.CONVN_WOBN(num_layers)
        else:
            model = conv.CONVN(num_layers)

    else:
        print("No model specified, use fc3")
        import models.fc as fc

        model = fc.FCN(3)

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
            if args.im == "xavier":
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif args.im == "kaiming_out":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif args.im == "kaiming_in":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif args.im == "haocheng":
                m.weight.data = torch.randn(m.weight.shape)
                m.weight.data /= torch.linalg.norm(m.weight, ord=2)
            else:
                print("No initialization method specified")

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
            score_dict = snip(model, trainloader)
            threshold = cal_threshold(score_dict, args.prune)
            apply_prune(model, score_dict, threshold)
        elif args.algorithm == "synflow":
            example_data, _ = next(iter(trainloader))
            iterations = 100
            for i in range(iterations):
                prune_ratio = args.prune / iterations * (i + 1)
                score_dict = synflow(model, example_data)
                threshold = cal_threshold(score_dict, prune_ratio)
                apply_prune(model, score_dict, threshold)
                if i != iterations - 1:
                    remove_mask(model)
        else:
            for name, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    prune.random_unstructured(m, name="weight", amount=args.prune)
                elif isinstance(m, nn.Linear):
                    prune.random_unstructured(m, name="weight", amount=args.prune)
        print(f"Pruned Sparsity: {cal_sparsity(model)}%")

    if args.restore != 0:
        print("restoring !!!!")
        if args.restore == 1:  # restore weight
            print("restore weight")
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    spec_norm = torch.linalg.norm(
                        m.weight.view(m.weight.shape[0], -1), ord=2
                    )
                    m.weight.data /= spec_norm
                    if args.prune != 0.0:
                        m.weight_orig.data /= spec_norm
                elif isinstance(m, nn.Linear):
                    spec_norm = torch.linalg.norm(m.weight, ord=2)
                    m.weight.data /= spec_norm
                    if args.prune != 0.0:
                        m.weight_orig.data /= spec_norm
        elif args.restore == 2:  # restore bn
            print("restore BN")
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    spec_norm = torch.linalg.norm(
                        m.weight.view(m.weight.shape[0], -1), ord=2
                    )
                    shape = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                    lv = spec_norm / torch.sqrt(0.5 * shape * m.weight.var())
                    bn_name = n.replace("conv", "bn")
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm2d):
                            print(f"{bn_name} not found")
                            continue
                        bn.weight.data /= lv
                        print(f"{bn_name} founded")
                        print(bn)
                    except:
                        print(f"{bn_name} not found")
                        pass
                elif isinstance(m, nn.Linear):
                    spec_norm = torch.linalg.norm(m.weight, ord=2)
                    lv = spec_norm / torch.sqrt(
                        0.5 * m.weight.shape[1] * m.weight.var()
                    )
                    bn_name = n.replace("fc", "bn")
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm1d):
                            print(f"{bn_name} not found")
                            continue
                        bn.weight.data /= lv
                        print(f"{bn_name} founded")
                        print(bn)
                    except:
                        print(f"{bn_name} not found")
                        pass
        elif args.restore == 3:  # restore both
            print("restoring BN+weight")
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    spec_norm = torch.linalg.norm(
                        m.weight.view(m.weight.shape[0], -1), ord=2
                    )
                    shape = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                    lv = spec_norm / torch.sqrt(0.5 * shape * m.weight.var())
                    m.weight.data /= spec_norm
                    if args.prune != 0.0:
                        m.weight_orig.data /= spec_norm
                    bn_name = n.replace("conv", "bn")
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm2d):
                            print(f"{bn_name} not found")
                            continue
                        bn.weight.data /= lv
                        print(f"{bn_name} founded")
                        print(bn)
                    except:
                        print(f"{bn_name} not found")
                        pass
                elif isinstance(m, nn.Linear):
                    spec_norm = torch.linalg.norm(m.weight, ord=2)
                    lv = spec_norm / torch.sqrt(
                        0.5 * m.weight.shape[1] * m.weight.var()
                    )
                    m.weight.data /= spec_norm
                    if args.prune != 0.0:
                        m.weight_orig.data /= spec_norm
                    bn_name = n.replace("fc", "bn")
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm1d):
                            print(f"{bn_name} not found")
                            continue
                        bn.weight.data /= lv
                        print(f"{bn_name} founded")
                        print(bn)
                    except:
                        print(f"{bn_name} not found")
                        pass
        elif args.restore == 4:  # restore both and move mean
            print("restoring BN+weight+mean")
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    mean = m.weight.data.mean()
                    m.weight.data -= mean
                    spec_norm = torch.linalg.norm(
                        m.weight.view(m.weight.shape[0], -1), ord=2
                    )
                    shape = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                    lv = spec_norm / torch.sqrt(0.5 * shape * m.weight.var())
                    m.weight.data /= spec_norm
                    if args.prune != 0.0:
                        m.weight_orig.data -= mean
                        m.weight_orig.data /= spec_norm
                    bn_name = n.replace("conv", "bn")
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm2d):
                            print(f"{bn_name} not found")
                            continue
                        bn.weight.data /= lv
                        print(f"{bn_name} founded")
                        print(bn)
                    except:
                        print(f"{bn_name} not found")
                        pass
                elif isinstance(m, nn.Linear):
                    mean = m.weight.data.mean()
                    m.weight.data -= mean
                    spec_norm = torch.linalg.norm(m.weight, ord=2)
                    lv = spec_norm / torch.sqrt(
                        0.5 * m.weight.shape[1] * m.weight.var()
                    )
                    m.weight.data /= spec_norm
                    if args.prune != 0.0:
                        m.weight_orig.data -= mean
                        m.weight_orig.data /= spec_norm
                    bn_name = n.replace("fc", "bn")
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm1d):
                            print(f"{bn_name} not found")
                            continue
                        bn.weight.data /= lv
                        print(f"{bn_name} founded")
                        print(bn)
                    except:
                        print(f"{bn_name} not found")
                        pass
        elif args.restore == 5:  # restore weight and mean
            print("restore weight")
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    mean = m.weight.data.mean()
                    m.weight.data -= mean
                    spec_norm = torch.linalg.norm(
                        m.weight.view(m.weight.shape[0], -1), ord=2
                    )
                    m.weight.data /= spec_norm
                    if args.prune != 0.0:
                        m.weight_orig.data -= mean
                        m.weight_orig.data /= spec_norm
                elif isinstance(m, nn.Linear):
                    mean = m.weight.data.mean()
                    m.weight.data -= mean
                    spec_norm = torch.linalg.norm(m.weight, ord=2)
                    m.weight.data /= spec_norm
                    if args.prune != 0.0:
                        m.weight_orig.data -= mean
                        m.weight_orig.data /= spec_norm

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
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
            torch.save(model.state_dict(), save_path + "/best.pth")

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
