#!/bin/sh
# $1 is the model
# $2 is the initial method
# $3 is the id number
# $4 is the restore flag
for i in $(seq 0.80 0.02 0.98)
do
	command="python cifar10.py -p $i -i $2 -s $3 -m $1 -r $4"
	echo $command
	$command
done
