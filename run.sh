num=$1
# res: bn, fc
python resnet_pwr.py --save logs/resnet_pwr_no.$num/ki_lr0.1s --lr 0.1 --prune 0.95 --restore 1 --im kaiming_in &
# res: bn, conv, fc
python resnet_pwr.py --save logs/resnet_pwr2_no.$num/ki_lr0.1s --lr 0.1 --prune 0.95 --restore 2 --im kaiming_in &
# res: fc
# python resnet_pwr.py --save logs/resnet_pwr3_no.$num/ki_lr0.1s --lr 0.1 --prune 0.95
# code changed
# res: none
python resnet_pwr.py --save logs/resnet_pwr4_no.$num/ki_lr0.1s --lr 0.1 --prune 0.95 &

