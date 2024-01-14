#!/bin/bash
# $1 is the model
# $2 is the initial method
for i in {1..5}
do
    sh core_cifar10.sh $1 $2 $i r1 &
done
