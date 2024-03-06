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
from data.imagenet import imagenet, DataLoaderX

import os, random, time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

use_ddp = "torch.nn.parallel.distributed"


def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def init_parallel(rank, world_size):
    if rank != 0:
        time.sleep(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )


# def init_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)


def __getattr(model, name):
    name_list = name.split(".")
    ret = model
    for n in name_list:
        if n.isdigit():
            ret = ret[int(n)]
        else:
            ret = getattr(ret, n)
    return ret


def train(
    trainset,
    testset,
    model,
    criterion,
    optimizer,
    scheduler,
    save_path,
    writer,
):
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    datasampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True
    )
    trainloader = DataLoaderX(
        trainset,
        batch_size=60,
        num_workers=4,
        pin_memory=True,
        sampler=datasampler,
    )
    testloader = DataLoaderX(
        testset,
        batch_size=64 * 4,
        num_workers=4,
        pin_memory=True,
    )
    # Training loop
    best_top1 = 0
    best_top5 = 0
    for epoch in tqdm.trange(100):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            # for i, data in tqdm.tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199 and writer is not None:
                writer.add_scalar(
                    "training loss", running_loss / 200, epoch * len(trainloader) + i
                )
                running_loss = 0.0
        scheduler.step()

        # Calculate accuracy on train and test sets
        if writer is not None:
            correct_top1 = 0
            correct_top5 = 0
            total_top1 = 0
            total_top5 = 0
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    # top-1
                    _, predicted = torch.max(outputs.data, 1)
                    test_loss += criterion(outputs, labels).item()
                    total_top1 += labels.size(0)
                    correct_top1 += (predicted == labels).sum().item()

                    # top-5
                    _, predicted = torch.topk(outputs.data, 5, 1)
                    total_top5 += labels.size(0)
                    for i in range(5):
                        correct_top5 += (predicted[:, i] == labels).sum().item()

            top1_accuracy = 100 * correct_top1 / total_top1
            top5_accuracy = 100 * correct_top5 / total_top5
            if top1_accuracy > best_top1:
                best_top1 = top1_accuracy
            if top5_accuracy > best_top5:
                best_top5 = top5_accuracy
                # torch.save(model.state_dict(), save_path + "/best.pth")

            writer.add_scalar("top-1", top1_accuracy, epoch)
            writer.add_scalar("top-5", top5_accuracy, epoch)
            writer.add_scalar("test loss", test_loss / len(testloader), epoch)

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    print("Finished Training")
    print("Best test accuracy: {:.2f}%".format(best))


def parallel_main(rank, world_size, args):

    init_parallel(rank, world_size)

    # Set up TensorBoard writer with log directory
    save_path = f"logs/imagenet/{args.model}_{args.alpha}_{args.beta}/{args.im}/{args.algorithm}/{args.prune:.2f}/r{args.restore}/no.{args.save}"
    if rank == 0:
        writer = SummaryWriter(log_dir=save_path)
    else:
        writer = None

    if args.model == "resnet50":
        from models.resnet_imagenet import resnet50

        model = resnet50(num_classes=1000)
    elif args.model == "resnet50_res":
        from models.resnet_imagenet_res import resnet50

        model = resnet50(num_classes=1000, alpha=args.alpha, beta=args.beta)
    else:
        raise ValueError("model not found")

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
                # print("No initialization method specified")
                pass
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
                # print("No initialization method specified")
                pass

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
            # import time
            # t_start = time.time()
            # example_data, _ = next(iter(trainloader))
            example_data = torch.randn(1, 3, 224, 224)
            sign_dict = linearize(model)
            iterations = 100
            # t_loop_start = time.time()
            # print("ready time: ", t_loop_start - t_start)
            for i in range(iterations):
                # t_loop_start = time.time()
                prune_ratio = args.prune / iterations * (i + 1)
                score_dict = synflow(model, example_data)
                # t_score = time.time()
                threshold = cal_threshold(score_dict, prune_ratio)
                # t_threshold = time.time()
                apply_prune(model, score_dict, threshold)
                # t_apply = time.time()
                if i != iterations - 1:
                    remove_mask(model)
                # t_remove = time.time()
                # print(
                #     f"iter {i+1}/{iterations}, score: {t_score-t_loop_start:.2f}, threshold: {t_threshold-t_score:.2f}, apply: {t_apply-t_threshold:.2f}, remove: {t_remove-t_apply:.2f}"
                # )
            nonlinearize(model, sign_dict)
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

    # model.to(device)
    [trainset, testset] = imagenet("~/autodl-tmp/imagenet")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 80], gamma=0.1
    )
    train(
        trainset,
        testset,
        model,
        criterion,
        optimizer,
        scheduler,
        save_path,
        writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch TinyImageNet Training")
    parser.add_argument("-s", "--save", help="save path", default="0")
    parser.add_argument("-p", "--prune", help="prune rate", type=float, default=0.0)
    parser.add_argument("-a", "--algorithm", help="prune algorithm", default="nonprune")
    parser.add_argument("-r", "--restore", help="restore type", type=int, default=0)
    parser.add_argument("-w", "--world_size", help="world size", type=int, default=2)
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
        default=1,
    )
    # only works in resnet_res
    parser.add_argument(
        "--beta",
        help="beta",
        type=float,
        default=1,
    )
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(
        parallel_main,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
