#!/bin/sh
bash run_cifar10_prune.sh conv3 kaiming_in $1
bash run_cifar10_prune.sh conv3 kaiming_out $1
bash run_cifar10_prune.sh conv3 xavier $1
# bash run_cifar10_prune.sh fc3 kaiming_in $1
# bash run_cifar10_prune.sh fc3 kaiming_out $1
# bash run_cifar10_prune.sh fc3 xavier $1
