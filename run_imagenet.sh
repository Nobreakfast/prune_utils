#!/bin/sh
# $1 is the save count
## non-pruning
# python imagenet.py -m resnet50 -i xavier -r 0 -s $1
# python imagenet.py -m resnet50 -i kaiming_in -r 0 -s $1
# python imagenet.py -m resnet50 -i kaiming_out -r 0 -s $1
# python imagenet.py -m resnet50 -i kaiming_in -r 1 -s $1
# python imagenet.py -m resnet50 -i kaiming_in -r 2 -s $1
# python imagenet.py -m resnet50 -i kaiming_in -r 3 -s $1
# python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 1.0 --beta 0.1 -s $1
# python imagenet.py -m resnet50_res -i kaiming_in -r 3 --alpha 1.0 --beta 0.1 -s $1
# python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 0.7 --beta 0.3 -s $1
# python imagenet.py -m resnet50_res -i kaiming_in -r 3 --alpha 0.7 --beta 0.3 -s $1
# python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 0.5 --beta 0.5 -s $1
# python imagenet.py -m resnet50_res -i kaiming_in -r 3 --alpha 0.5 --beta 0.5 -s $1

## pruning
for i in $(seq 0.85 0.10 0.98)
do
    # python imagenet.py -m resnet50 -i xavier -r 0 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 0 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_out -r 0 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 5 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 2 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 4 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 1.0 --beta 0.1 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 1.0 --beta 0.1 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 0.7 --beta 0.3 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 0.7 --beta 0.3 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 0.5 --beta 0.5 -a uniform -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 0.5 --beta 0.5 -a uniform -p $i -s $1

    # python imagenet.py -m resnet50 -i xavier -r 0 -a synflow -p $i -s $1
    python imagenet.py -m resnet50 -i kaiming_in -r 0 -a synflow -p $i -s $1 -w 8
    # python imagenet.py -m resnet50 -i kaiming_out -r 0 -a synflow -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 5 -a synflow -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 2 -a synflow -p $i -s $1
    python imagenet.py -m resnet50 -i kaiming_in -r 4 -a synflow -p $i -s $1 -w 8
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 1.0 --beta 0.1 -a synflow -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 1.0 --beta 0.1 -a synflow -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 0.7 --beta 0.3 -a synflow -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 0.7 --beta 0.3 -a synflow -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 0.5 --beta 0.5 -a synflow -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 0.5 --beta 0.5 -a synflow -p $i -s $1

    # python imagenet.py -m resnet50 -i xavier -r 0 -a snip -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 0 -a snip -p $i -s $1 -w 8
    # python imagenet.py -m resnet50 -i kaiming_out -r 0 -a snip -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 5 -a snip -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 2 -a snip -p $i -s $1
    # python imagenet.py -m resnet50 -i kaiming_in -r 4 -a snip -p $i -s $1 -w 8
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 1.0 --beta 0.1 -a snip -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 1.0 --beta 0.1 -a snip -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 0.7 --beta 0.3 -a snip -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 0.7 --beta 0.3 -a snip -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 0 --alpha 0.5 --beta 0.5 -a snip -p $i -s $1
    # python imagenet.py -m resnet50_res -i kaiming_in -r 4 --alpha 0.5 --beta 0.5 -a snip -p $i -s $1
done