num=$1
python resnet3.py --restore 0.125 --save ./logs/resnet3_res_no.$num/km_lr0.1s &
python resnet3.py --save ./logs/resnet3_no.$num/km_lr0.1s