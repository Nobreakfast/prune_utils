#!/bin/sh
# $1 is the count
## pruning
# for i in $(seq 0.60 0.05 0.98)
# do
#     python cifar10.py -m resnet20 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 1
#     python cifar10.py -m resnet20 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 2
#     python cifar10.py -m resnet20 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 3
#     python cifar10.py -m resnet20 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 4

#     python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 1
#     python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 2
#     python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 3
#     python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 4

#     python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 1
#     python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 2
#     python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 3
#     python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 4

# done

for i in $(seq 0.60 0.05 0.98)
    python cifar10.py -m resnet20 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 3 &
    python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 3 &
    python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 3
    python cifar10.py -m resnet20 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 4 &
    python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 4 &
    python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a resynflow -p $i -s $1 --ablation 4
done