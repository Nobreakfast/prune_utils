#!/bin/bash
# $1 is the model
# $2 is the initial method
# $3 is the restore
for i in {1..5}
do
    python cifar10.py -i $2 -s $i -m $1 -r $3 &
done
