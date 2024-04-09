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
from prune_utils.repair import repair_model
from prune_utils.initial import initialization
from data.imagenet import imagenet
from data.core.dataloader import DataLoaderX

import os, random, time, datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

use_ddp = "torch.nn.parallel.distributed"


def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def init_parallel(rank, world_size, port):
    if rank != 0:
        time.sleep(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=100),
    )


def parallel_main(rank, world_size, port, trainset, testset, save_path, model, args):
    init_parallel(rank, world_size, port)

    train(trainset, testset, save_path, model, args)


def train(trainset, testset, save_path, model, args):
    rank = dist.get_rank()
    if rank == 0:
        writer = SummaryWriter(log_dir=save_path)
    else:
        writer = None
        testset = None

    if args.im == "lsuv":
        from lsuv import lsuv_with_dataloader

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = lsuv_with_dataloader(model, trainloader, device=device)
    elif args.im == "lipz":
        initialization(model, "kaiming_in")
        repair_model(model, 1)
    elif args.im == "rei":
        initialization(model, "kaiming_in")
        repair_model(model, 3)
    else:
        initialization(model, args.im)

    weight_dict = get_weight(model)

    if args.restore != 0:
        print("restoring !!!!")
        repair_model(model, args.restore)

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
            trainloader = DataLoaderX(
                trainset,
                batch_size=128,
                num_workers=1,
                pin_memory=False,
                shuffle=True,
            )
            device = torch.device(
                f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            )
            model = model.to(device)
            score_dict = snip(model, trainloader)
            threshold = cal_threshold(score_dict, args.prune)
            apply_prune(model, score_dict, threshold)
            model = model.to(torch.device("cpu"))
        elif args.algorithm == "synflow":
            device = torch.device(
                f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            )
            model = model.to(device)
            example_data = torch.randn(1, 3, 224, 224)
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
            device = torch.device(
                f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            )
            model = model.to(device)
            example_data = torch.randn(1, 3, 224, 224)
            sign_dict = linearize(model)
            iterations = 10
            for i in range(iterations):
                for name, m in model.named_modules():
                    if isinstance(m, nn.Conv2d):
                        sn = torch.linalg.norm(
                            m.weight.view(m.weight.shape[0], -1), ord=2
                        ).item()
                        m.weight.data /= sn
                    elif isinstance(m, nn.Linear):
                        sn = torch.linalg.norm(m.weight, ord=2).item()
                        m.weight.data /= sn
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
        # for name, m in model.named_modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         print(f"{name} sparsity: {cal_sparsity(m)}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.ablation != 0:
        if args.ablation == 1:
            print("reinit")
            mask_dict = get_mask(model)
            remove_mask(model)
            initialization(model, "kaiming_in")
            apply_prune(model, mask_dict, 1)
        elif args.ablation == 2:
            print("init")
            lw_dict = get_lw_sparsity(model)
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight_mask.data = torch.ones_like(module.weight)
                    module.weight_orig.data = weight_dict[name]
                    module.weight.data = module.weight_orig.data * module.weight_mask
                    prune.l1_unstructured(module, name="weight", amount=lw_dict[name])
        elif args.ablation == 3:
            print("shuffle")
            lw_dict = get_lw_sparsity(model)
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight_orig.data = weight_dict[name]
                    module.weight.data = module.weight_orig.data * module.weight_mask
        print(f"Ablation {args.ablation} Sparsity: {cal_sparsity(model)}")

    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    world_size = dist.get_world_size()
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 80], gamma=0.1
    )

    train_datasampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True
    )
    trainloader = DataLoaderX(
        trainset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        sampler=train_datasampler,
    )
    # test_datasampler = torch.utils.data.distributed.DistributedSampler(
    #     testset, shuffle=True
    # )
    if rank == 0:
        testloader = DataLoaderX(
            testset,
            batch_size=128 * 4,
            num_workers=16,
            pin_memory=True,
        )
    # Training loop
    best_top1 = 0
    best_top5 = 0
    for epoch in tqdm.trange(90, leave=False):
        running_loss = 0.0
        model.train()
        trainloader.sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader, 0):
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

        if rank == 0:
            # Calculate accuracy on train and test sets
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
                torch.save(model.state_dict(), save_path + "/best.pth")
            if top5_accuracy > best_top5:
                best_top5 = top5_accuracy
            model.train()

        if rank == 0:
            writer.add_scalar("top-1", top1_accuracy, epoch)
            writer.add_scalar("top-5", top5_accuracy, epoch)
            writer.add_scalar("test loss", test_loss / len(testloader), epoch)

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch TinyImageNet Training")
    parser.add_argument("-s", "--save", help="save path", default="0")
    parser.add_argument("-p", "--prune", help="prune rate", type=float, default=0.0)
    parser.add_argument("-a", "--algorithm", help="prune algorithm", default="nonprune")
    parser.add_argument("-r", "--restore", help="restore type", type=int, default=0)
    parser.add_argument("-w", "--world_size", help="world size", type=int, default=2)
    parser.add_argument("-t", "--port", help="port", default="12345")
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
    parser.add_argument(
        "--ablation",
        help="ablation",
        type=int,  # 1: reinit, 2: init, 3: shuffle, 4: invert
        default=0,
    )
    args = parser.parse_args()

    # Set up TensorBoard writer with log directory
    save_path = f"logs/imagenet/{args.model}_{args.alpha}_{args.beta}/{args.im}/{args.algorithm}/{args.prune:.2f}/r{args.restore}/no.{args.save}"
    os.system(f"mkdir -p {save_path}")

    [trainset, testset] = imagenet("/home/allen/Data/imagenet")

    if args.model == "resnet50":
        from models.resnet_imagenet import resnet50

        model = resnet50(num_classes=1000)
    else:
        raise ValueError("model not found")

    world_size = args.world_size
    mp.spawn(
        parallel_main,
        args=(world_size, args.port, trainset, testset, save_path, model, args),
        nprocs=world_size,
        join=True,
    )
