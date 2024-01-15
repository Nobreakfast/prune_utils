#!/bin/bash
for i in $(seq 0.80 0.02 0.98)
do
	if [ "$1" = "r1" ]; then
		echo "python resnet_wobn.py -p $i -i $2 -s logs/resnet_wobn/r1_${2}_p${i}_no.$3 -r 1"
		python resnet_wobn.py -p $i -i $2 -s logs/resnet_wobn/r1_${2}_p${i}_no.$3 -r 1
	elif [ "$1" = "bl" ]; then
		echo "python resnet_wobn.py -p $i -i $2 -s logs/resnet_wobn/bl_${2}_p${i}_no.$3"
		python resnet_wobn.py -p $i -i $2 -s logs/resnet_wobn/bl_${2}_p${i}_no.$3
	fi
done
