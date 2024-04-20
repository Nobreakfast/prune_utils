#!/bin/sh
# $1 is the save count
## pruning

for i in $(seq 0.95 0.05 0.98)
do
    # ## cifar10
    # uniform
    # python cifar10.py -m resnet20 -i kaiming_in -r 12 -a uniform -p $i -s $1 &
    # python cifar10.py -m resnet20 -i kaiming_out -r 14 -a uniform -p $i -s $1 &
    # python cifar10.py -m resnet20 -i xavier -r 16 -a uniform -p $i -s $1 
    # # synflow
    # python cifar10.py -m resnet20 -i kaiming_in -r 12 -a synflow -p $i -s $1 &
    # python cifar10.py -m resnet20 -i kaiming_out -r 14 -a synflow -p $i -s $1 &
    # python cifar10.py -m resnet20 -i xavier -r 16 -a synflow -p $i -s $1
    # # snip
    # python cifar10.py -m resnet20 -i kaiming_in -r 12 -a snip -p $i -s $1 &
    # python cifar10.py -m resnet20 -i kaiming_out -r 14 -a snip -p $i -s $1 &
    # python cifar10.py -m resnet20 -i xavier -r 16 -a snip -p $i -s $1
    # # reynflow
    # python cifar10.py -m resnet20 -i kaiming_in -r 12 -a reynflow -p $i -s $1 &
    # # python cifar10.py -m resnet20 -i kaiming_out -r 14 -a reynflow -p $i -s $1 &
    # # python cifar10.py -m resnet20 -i xavier -r 16 -a reynflow -p $i -s $1

    ## cifar100
    # uniform
    python cifar100.py -m vgg16_bn -i kaiming_in -r 12 -a uniform -p $i -s $1 &
    python cifar100.py -m vgg16_bn -i kaiming_out -r 14 -a uniform -p $i -s $1 &
    python cifar100.py -m vgg16_bn -i xavier -r 16 -a uniform -p $i -s $1
    # synflow
    python cifar100.py -m vgg16_bn -i kaiming_in -r 12 -a synflow -p $i -s $1 &
    python cifar100.py -m vgg16_bn -i kaiming_out -r 14 -a synflow -p $i -s $1 &
    python cifar100.py -m vgg16_bn -i xavier -r 16 -a synflow -p $i -s $1
    # snip
    python cifar100.py -m vgg16_bn -i kaiming_in -r 12 -a snip -p $i -s $1 &
    python cifar100.py -m vgg16_bn -i kaiming_out -r 14 -a snip -p $i -s $1 &
    python cifar100.py -m vgg16_bn -i xavier -r 16 -a snip -p $i -s $1
    # reynflow
    python cifar100.py -m vgg16_bn -i kaiming_in -r 12 -a reynflow -p $i -s $1 &
    # python cifar100.py -m vgg16_bn -i kaiming_out -r 14 -a reynflow -p $i -s $1 &
    # python cifar100.py -m vgg16_bn -i xavier -r 16 -a reynflow -p $i -s $1

    ## tiny-imagenet
    # uniform
    python tinyimagenet.py -m resnet18 -i kaiming_in -r 12 -a uniform -p $i -s $1 
    python tinyimagenet.py -m resnet18 -i kaiming_out -r 14 -a uniform -p $i -s $1 
    python tinyimagenet.py -m resnet18 -i xavier -r 16 -a uniform -p $i -s $1 
    # synflow
    python tinyimagenet.py -m resnet18 -i kaiming_in -r 12 -a synflow -p $i -s $1 
    python tinyimagenet.py -m resnet18 -i kaiming_out -r 14 -a synflow -p $i -s $1 
    python tinyimagenet.py -m resnet18 -i xavier -r 16 -a synflow -p $i -s $1
    # snip
    python tinyimagenet.py -m resnet18 -i kaiming_in -r 12 -a snip -p $i -s $1 
    python tinyimagenet.py -m resnet18 -i kaiming_out -r 14 -a snip -p $i -s $1 
    python tinyimagenet.py -m resnet18 -i xavier -r 16 -a snip -p $i -s $1
    # reynflow
    python tinyimagenet.py -m resnet18 -i kaiming_in -r 12 -a reynflow -p $i -s $1 
    # python tinyimagenet.py -m resnet18 -i kaiming_out -r 14 -a reynflow -p $i -s $1 &
    # python tinyimagenet.py -m resnet18 -i xavier -r 16 -a reynflow -p $i -s $1
done

for i in $(seq 0.80 0.05 0.90)
do
    ## tiny-imagenet
    # synflow
    python tinyimagenet.py -m resnet18 -i kaiming_in -r 12 -a synflow -p $i -s $1 
    python tinyimagenet.py -m resnet18 -i kaiming_out -r 14 -a synflow -p $i -s $1 
    python tinyimagenet.py -m resnet18 -i xavier -r 16 -a synflow -p $i -s $1
done
