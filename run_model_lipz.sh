#!/bin/sh
# model: fc3, conv3
# init method: kaiming_in, kaiming_out, xavier
# $1 prune method: rand, randn, snip, synflow, uniform
model=("fc3" "conv3")
init_method=("kaiming_in" "kaiming_out" "xavier")
for m in ${model[@]}
do
    for i in ${init_method[@]}
    do
        for p in $(seq 0.00 0.02 0.98)
        do
            for r in {1..3}
            do
                for s in {1..10}
                do
                    command="python3 cifar10_lipz.py --model $m --im $i --algorithm $1 --prune $p --restore $r --save $s"
                    $command
                done
            done
        done
    done
done


model=("fc3_wobn" "conv3_wobn")
for m in ${model[@]}
do
    for i in ${init_method[@]}
    do
        for p in $(seq 0.00 0.02 0.98)
        do
            for s in {1..10}
            do
                command="python3 cifar10_lipz.py --model $m --im $i --algorithm $1 --prune $p --restore 1 --save $s"
                $command
            done
        done
    done
done
 