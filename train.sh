# nohup python train.py --gpu_ids 0,1 --batch_size 80 --experiment_name batch80 --pretrain_flownet None > results/train.log 2>&1 &
nohup python train.py --gpu_ids 0,1 --batch_size 32 --workers 32 --experiment_name batch32 > results/train.log 2>&1 &