#!/bin/bash
# $1 is the initial method
for i in {1..5}
do
    python cifar10.py -i $1 -s $i -m fc3 -r 2 &
    python cifar10.py -i $1 -s $i -m conv3 -r 2 &
    python cifar10.py -i $1 -s $i -m fc3 -r 3 &
    python cifar10.py -i $1 -s $i -m conv3 -r 3
done