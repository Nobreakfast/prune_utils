#!/bin/sh
# $1 is the save count
## non-pruning
# python tinyimagenet.py -m resnet18 -i xavier -r 0 -s $1
# python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -s $1
# python tinyimagenet.py -m resnet18 -i kaiming_out -r 0 -s $1
# python tinyimagenet.py -m resnet18 -i kaiming_in -r 1 -s $1
# python tinyimagenet.py -m resnet18 -i kaiming_in -r 2 -s $1
# python tinyimagenet.py -m resnet18 -i kaiming_in -r 3 -s $1
# python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 1.0 --beta 0.1 -s $1
# python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 1.0 --beta 0.1 -s $1
# python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 0.7 --beta 0.3 -s $1
# python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 0.7 --beta 0.3 -s $1
# python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 0.5 --beta 0.5 -s $1
# python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 0.5 --beta 0.5 -s $1

## pruning
i=$2
python tinyimagenet.py -m resnet18 -i xavier -r 0 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_out -r 0 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 1 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 2 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 3 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 1.0 --beta 0.1 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 1.0 --beta 0.1 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 0.7 --beta 0.3 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 0.7 --beta 0.3 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 0.5 --beta 0.5 -a uniform -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 0.5 --beta 0.5 -a uniform -p $i -s $1

python tinyimagenet.py -m resnet18 -i xavier -r 0 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_out -r 0 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 1 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 2 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 3 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 1.0 --beta 0.1 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 1.0 --beta 0.1 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 0.7 --beta 0.3 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 0.7 --beta 0.3 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 0.5 --beta 0.5 -a synflow -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 0.5 --beta 0.5 -a synflow -p $i -s $1

python tinyimagenet.py -m resnet18 -i xavier -r 0 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 0 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_out -r 0 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 1 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 2 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18 -i kaiming_in -r 3 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 1.0 --beta 0.1 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 1.0 --beta 0.1 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 0.7 --beta 0.3 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 0.7 --beta 0.3 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 0 --alpha 0.5 --beta 0.5 -a snip -p $i -s $1
python tinyimagenet.py -m resnet18_res -i kaiming_in -r 3 --alpha 0.5 --beta 0.5 -a snip -p $i -s $1
