#!/bin/sh
# $1 is the model
# $2 is the initial method
# $3 is the restore
for i in {1..5}
do
    sh core_cifar10_prune.sh $1 $2 $i $3 &
done
