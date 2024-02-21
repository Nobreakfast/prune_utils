#!/bin/sh
# $1 is the save count
## non-pruning
# wobn
python cifar100.py -m vgg16 -i xavier -r 0 -s $1
python cifar100.py -m vgg16 -i kaiming_in -r 0 -s $1
python cifar100.py -m vgg16 -i kaiming_out -r 0 -s $1
python cifar100.py -m vgg16 -i kaiming_in -r 1 -s $1
#bn
python cifar100.py -m vgg16_bn -i xavier -r 0 -s $1
python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -s $1
python cifar100.py -m vgg16_bn -i kaiming_out -r 0 -s $1
python cifar100.py -m vgg16_bn -i kaiming_in -r 1 -s $1
python cifar100.py -m vgg16_bn -i kaiming_in -r 2 -s $1
python cifar100.py -m vgg16_bn -i kaiming_in -r 3 -s $1

## pruning
for i in $(seq 0.80 0.05 0.98)
do
    # wobn
    # uniform
    python cifar100.py -m vgg16 -i xavier -r 0 -a uniform -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_in -r 0 -a uniform -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_out -r 0 -a uniform -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_in -r 5 -a uniform -p $i -s $1
    # synflow
    python cifar100.py -m vgg16 -i xavier -r 0 -a synflow -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_in -r 0 -a synflow -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_out -r 0 -a synflow -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_in -r 5 -a synflow -p $i -s $1
    # snip
    python cifar100.py -m vgg16 -i xavier -r 0 -a snip -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_in -r 0 -a snip -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_out -r 0 -a snip -p $i -s $1
    python cifar100.py -m vgg16 -i kaiming_in -r 5 -a snip -p $i -s $1

    # bn
    # uniform
    python cifar100.py -m vgg16_bn -i xavier -r 0 -a uniform -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a uniform -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_out -r 0 -a uniform -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 5 -a uniform -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 2 -a uniform -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 4 -a uniform -p $i -s $1
    # synflow
    python cifar100.py -m vgg16_bn -i xavier -r 0 -a synflow -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a synflow -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_out -r 0 -a synflow -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 5 -a synflow -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 2 -a synflow -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 4 -a synflow -p $i -s $1
    # snip
    python cifar100.py -m vgg16_bn -i xavier -r 0 -a snip -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 0 -a snip -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_out -r 0 -a snip -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 5 -a snip -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 2 -a snip -p $i -s $1
    python cifar100.py -m vgg16_bn -i kaiming_in -r 4 -a snip -p $i -s $1
done