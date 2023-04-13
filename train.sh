# nohup python train.py --gpu_ids 0,1 --batch_size 80 --experiment_name batch80 --pretrain_flownet None > results/train.log 2>&1 &
python train.py --batch_size 32 --workers 32
python train.py --gpu_ids 0,1 --batch_size 32 --workers 32 --experiment_name batch32
python train.py --gpu_ids 2,3 --batch_size 32 --workers 32 --experiment_name transformer_32 --transformer
python train.py --gpu_ids 4,5 --batch_size 32 --workers 32 --experiment_name dense_32 --transformer --dense_connect --pretrain results/dense_32/checkpoints/081.pth

python train.py --gpu_ids 0,1,2,3 --batch_size 32 --workers 32 --experiment_name seq32 --transformer --seq2seq

#----2023/4/8
python train.py --gpu_ids 1 --experiment_name full_transformer --transformer --seq2seq --batch_size 16 --workers 4
#接着080.pth继续train
python train.py --gpu_ids 1 --data_dir ../Visual-Selective-VIO/data --experiment_name full_transformer --batch_size 16 --pretrain ./results/full_transformer/checkpoints/080.pth
python test.py --gpu_ids 0 --data_dir ../Visual-Selective-VIO/data --experiment_name test_encoder_decoder --model ./results/full_transformer/checkpoints/045.pth

#修改为2023/4/12开会时所说的加时序的结果，跑一下看看结果！！！！try_have_time_series  in test.py ; in train.py is_first is all true 
python train.py --gpu_ids 0 --data_dir ../Visual-Selective-VIO/data --experiment_name try_have_time_series --batch_size 16 --transformer --seq2seq

