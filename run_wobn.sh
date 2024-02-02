#!/bin/sh
bash run_cifar10_prune.sh conv3_wobn kaiming_in $1
bash run_cifar10_prune.sh conv3_wobn kaiming_out $1
bash run_cifar10_prune.sh conv3_wobn xavier $1
bash run_cifar10_prune.sh fc3_wobn kaiming_in $1
bash run_cifar10_prune.sh fc3_wobn kaiming_out $1
bash run_cifar10_prune.sh fc3_wobn xavier $1
