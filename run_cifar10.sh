#!/bin/bash
# $1 is the initial method
# $2 is the restore
for i in {1..5}
do
    python cifar10.py -i $1 -s $i -m fc3 -r $2 &
    python cifar10.py -i $1 -s $i -m fc3_wobn -r $2 &
    python cifar10.py -i $1 -s $i -m conv3 -r $2 &
    python cifar10.py -i $1 -s $i -m conv3_wobn -r $2 &
done
