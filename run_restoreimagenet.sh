#!/bin/sh
# $1 is the save count
## pruning

for i in $(seq 0.85 0.10 0.98)
do
    # uniform
    python imagenet.py -m resnet50 -i kaiming_in -r 12 -a uniform -p $i -s $1 &
    python imagenet.py -m resnet50 -i kaiming_out -r 14 -a uniform -p $i -s $1 &
    python imagenet.py -m resnet50 -i xavier -r 16 -a uniform -p $i -s $1 
    # reynflow
    python imagenet.py -m resnet50 -i kaiming_in -r 12 -a reynflow -p $i -s $1 &
    python imagenet.py -m resnet50 -i kaiming_out -r 14 -a reynflow -p $i -s $1 &
    python imagenet.py -m resnet50 -i xavier -r 16 -a reynflow -p $i -s $1
done
