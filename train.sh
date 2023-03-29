# nohup python train.py --gpu_ids 0,1 --batch_size 80 --experiment_name batch80 --pretrain_flownet None > results/train.log 2>&1 &
python train.py --gpu_ids 0,1 --batch_size 32 --workers 32 --experiment_name batch32
python train.py --gpu_ids 2,3 --batch_size 32 --workers 32 --experiment_name transformer_32 --transformer
python train.py --gpu_ids 4,5 --batch_size 32 --workers 32 --experiment_name dense_32 --transformer --dense_connect --pretrain results/dense_32/checkpoints/081.pth

python train.py --gpu_ids 0,1,2,3 --batch_size 32 --workers 32 --experiment_name seq32 --transformer --seq2seq
